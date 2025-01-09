from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableBranch
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.globals import set_debug
# set_debug(True)
from pydantic import BaseModel, Field
from indexing import Indexing
from retrieve import Retrieve
from query_translation import QueryTranslation
import json
import sys
from csv import DictReader

import os
from dotenv import load_dotenv
load_dotenv()

class Output(BaseModel):
    """
    Represents the output schema/structure
    """
    database_name:str = Field(description="Database Name")
    schema_name:str = Field(description="Schema Name")
    column_name:str = Field(description="Column name")
    business_concept: str = Field(description="business concept")
    business_description: str = Field(description="business description")

class RAG:
    """
    Retrieval Augmented Generation
    """
    def __init__(self):
        self.query_translate_obj = QueryTranslation()
        self.indexing_obj = Indexing()
        self.output_parser = JsonOutputParser(pydantic_object=Output)
        self.format_instructions = self.output_parser.get_format_instructions()

    def prompt(self, input_data):
        """
        represents system, user prompt
        """
        print("constructing prompt")
        self.system_message =""" A chat between human and a helpful assistant that automatically creates data catalogs.
                                The assitant provides business concepts and business description corresponding to the human's input.
                                The assistant follows these rules for the business concept provided:
                                - is short containing only one or two words,
                                - is usually a higher level and more generic concept than the column name,
                                - does not reuse the column name with the word 'column' added after it,
                               The  assistant follows these rules for the business descriptions provided:
                               - single sentence no longer than twenty words,
                               - does not start with the word 'A column',
                               - does not reuse the table name,
                               - does not mention that the column is in the table name,
                               - does not contain examples and does not mention 'such as',
                               - is never the same as the business concept.
                            """
        self.user_message = f"""
                            Input: {input_data['input_data']}\n
                            The following examples might be helpful to the assistant:\n 
                            Retrieved_documents: {input_data['kb_retrieved']}\n
                            Relevant_Columns: {input_data['relevant_columns']}
                            These are only examples and should not be reused as-is unless the column name is a perfect match
                            and the description follows all the rules.\n
         
                           """
        self._prompt = PromptTemplate(
            input_variables=['input_data', 'kb_retrieved', 'relevant_columns'],
            output_variables = ["business_concept", "business_description"],
            template = self.system_message + "\n" + self.user_message
        )
        self._prompt = f"{self._prompt} \n{self.output_parser.get_format_instructions()}"
        print(self._prompt)
        return self._prompt

    def generate(self, prompt):
        print("LLM Response Generation")
        #llm
        llm = ChatGoogleGenerativeAI(
            model = "gemini-1.5-flash",
            temperature=0
        )
        return llm

    def chain(self):
        """chain"""
        print("Starting RAG workflow")
        fetch_branch = RunnableBranch(
            (lambda x: x['input_type'] == 'csv', lambda x: self.query_translate_obj.fetch_data_from_csv(x)),
            (lambda x: x['input_type'] == 'db', lambda x: self.query_translate_obj.fetch_data_from_csv(x)),
            (lambda x: "Invalid input type")
        )

        kb_indexing = RunnableLambda(lambda x: self.indexing_obj.load(x)) |\
                      RunnableLambda(lambda x: self.indexing_obj.split()) |\
                      RunnableLambda(lambda x: self.indexing_obj.create_vector_store("faiss_demo"))
        
        json_dump = RunnableLambda(lambda x: json.dump(x, open('chain1_out.json', 'w')))

        chain1 = (fetch_branch | kb_indexing | json_dump)

        chain1.invoke({
            "input_type": "csv",
            "input_details": {
                "file_path": "metadata.csv",
            },
            "kb_file": "kb_file.pdf"
            }
        )
        input_dict = json.load(open('chain1_out.json'))

        self.retrieve_obj = Retrieve(input_dict)
        retrieve = RunnableLambda(lambda x: self.retrieve_obj.retrieve(x))
        save_output_in_vectordb = RunnableLambda(lambda x: self.indexing_obj.save_output_in_vectordb(x))
        
        with open('metadata.csv') as f:
            csv_reader = DictReader(f)
            # next(csv_reader)
            _input_data = list(csv_reader)
        print(_input_data)

        chain2 = (retrieve | self.prompt | self.generate | self.output_parser | save_output_in_vectordb)
        result = chain2.batch(_input_data)
        print(result)

if __name__ == "__main__":
    rag_obj = RAG()
    rag_obj.chain()