from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import json

class Indexing:

    def load(self, input_dict):
        print("loading file")
        self.input_dict = input_dict
        file_path = input_dict['kb_file']
        loader = PyPDFLoader(file_path)
        self.documents = loader.load()
        return self.documents

    def split(self, chunk_size=1000, chunk_overlap=100):
        print("splitting file")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.chunks = text_splitter.split_documents(self.documents)
        return self.chunks

    def create_vector_store(self, vector_db):
        print("creating vector db and embeddings")
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normaize_embeddings': False}
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2",
                                           model_kwargs=model_kwargs,
                                           encode_kwargs=encode_kwargs)
        try:
            self.faiss_db = FAISS.from_documents(self.chunks, embeddings)
            self.faiss_db.save_local(vector_db)
            self.input_dict.update({"vector_db": vector_db})
            return self.input_dict
        except Exception as e:
            print(e)
    
    def save_output_in_vectordb(self, output):
        print("##################")
        output_string = json.dumps(output)
        print(f"!!!!!!!!!!!!!!{[output_string]}")
        print("creating output vector db and embeddings")
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normaize_embeddings': False}
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2",
                                           model_kwargs=model_kwargs,
                                           encode_kwargs=encode_kwargs)
        try:
            self.faiss_output_db = FAISS.from_texts([output_string], embeddings)
            self.faiss_output_db.save_local("faiss_output_db")
            self.input_dict.update({"output_vector_db": "faiss_output_db"})
            return self.input_dict
        except Exception as e:
            print(e)

    

