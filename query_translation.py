from langchain_community.document_loaders import CSVLoader
import pandas as pd

class QueryTranslation:
    def fetch_data_from_csv(self, input_dict):
        print("fectching from csv")
        self.input_dict = input_dict
        df = pd.read_csv(f"{input_dict['input_details']['file_path']}")
        self.input_dict.update({"data": self.input_dict['input_details']['file_path']})
        return self.input_dict
    
    def fetch_data_from_db(self, input_dict):
        pass

    def preprocess_fetched_data(self):
        pass