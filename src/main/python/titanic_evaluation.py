"""
A class that handles model evaluation for the Titanic dataset.
"""

import pandas as pd


class TitanicEvaluation:
    """
    A class that handles model evaluation for the Titanic dataset.

    Parameters:
    - test_result_path (str): The file path to the test result.

    Methods:
    - evaluate_model(trained_model, preprocessed_test_data): Evaluates the
      trained models using the preprocessed test data.
    """

    def __init__(self, test_result_path):
        self.test_result_path = test_result_path

    def evaluate_model(self, trained_model, preprocessed_test_data):
        """
        Evaluates the trained model using the preprocessed test data.

        Parameters:
        trained_model (dict): A dictionary containing the trained models.
        preprocessed_test_data (DataFrame): The preprocessed test data.

        Returns:
        dict: A dictionary containing the evaluation results for each model.
        """
        x_test = preprocessed_test_data.drop(["Survived", "PassengerId"], axis=1)
        test_result = pd.read_csv(self.test_result_path)
        y_test = test_result["Survived"]

        evaluation_results = {}
        for model_name, model in trained_model.items():
            evaluation_results[model_name] = model.score(x_test, y_test)

        return evaluation_results
