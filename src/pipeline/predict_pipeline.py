import os
import sys
from pathlib import Path

import traceback
import numpy as np
import pandas as pd

from src.logger import logging
from src.exception import CustomException
from src.utils import load_obj
import os

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(BASE_DIR, '..', '..', 'artifacts', 'model.pkl')
            preprocessor_path = os.path.join(BASE_DIR, '..', '..', 'artifacts', 'preprocessor.pkl')
            model = load_obj(file_path = model_path)
            preprocessor = load_obj(file_path = preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    # This class will be responsible for mapping the values from html file to backend
    def __init__(
            self,
            gender:str,
            race_ethnicity:str,
            parental_level_of_education:str,
            lunch:str,
            test_preparation_course:str,
            reading_score:float,
            writing_score:float
    ):
        self.gender = gender
        self.race_ethnicity = race_ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race/ethnicity": [self.race_ethnicity],
                "parental level of education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test preparation course": [self.test_preparation_course],
                "reading score": [self.reading_score],
                "writing score": [self.writing_score]
            }

            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            print(f"Prediction failed: {str(e)}")
            traceback.print_exc()
            raise CustomException(e, sys)