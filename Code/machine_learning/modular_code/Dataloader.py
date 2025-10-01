import pandas as pd

import numpy as np

class Dataloader:
    def __init__(self,base_path="../../dataset-source/embedding-dataset/software-metrics/"):
        self.base_path = base_path
    
    def load_dataset(self):
        """ Loading all four datasets """
        
        datasets = {
            'LONG_METHOD' : pd.read_csv(f"{self.base_path}LongMethod_code_metrics_values.csv"),
            'GOD_CLASS': pd.read_csv(f"{self.base_path}GodClass_code_metrics_values.csv"),
            'FEATURE_ENVY': pd.read_csv(f"{self.base_path}FeatureEnvy_code_metrics_values.csv"),
            'DATA_CLASS': pd.read_csv(f"{self.base_path}DataClass_code_metrics_values.csv")
        }
        
        return datasets
    
    def preprocess_dataset(self,df, columns_to_drop = ['tcc', 'lcc']):
        
        df_clean = df.copy()
        