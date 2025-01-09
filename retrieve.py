from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

class Retrieve:

    def __init__(self, input_dict):
        self.input_dict = input_dict
        self.faiss_db = input_dict['vector_db']
    def retrieve(self, input_data):
        print("fectching relevant docs")
        input_data = "\n".join(f"'{key}' : '{val}'" for key,val in input_data.items() if key!="ROW_ID")
        print(f"@@@@@@{input_data}")
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normaize_embeddings': False}
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2",
                                           model_kwargs=model_kwargs,
                                           encode_kwargs=encode_kwargs)
        try:
            faissdb = FAISS.load_local(self.faiss_db, embeddings, allow_dangerous_deserialization=True)
            similar_docs = faissdb.similarity_search(input_data, k=1)

            faiss_output_db = FAISS.load_local('faiss_output_db', embeddings, allow_dangerous_deserialization=True)
            similar_columns = faiss_output_db.similarity_search(input_data, k=1)
            print(similar_columns)

            return {"kb_retrieved": similar_docs[0].page_content, "input_data": input_data, "relevant_columns": similar_columns[0].page_content}   
        except Exception as e:
            print(e)    
