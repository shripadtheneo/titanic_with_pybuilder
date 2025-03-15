"""
This module contains unit tests for the TitanicTraining class.
"""
import os
import pytest

from titanic_preprocessing import TitanicPreprocessing
from titanic_training import TitanicTraining
from titanic_evaluation import TitanicEvaluation


class TestTitanicTraining:
    """
    This class contains unit tests for the TitanicTraining class.
    """

    @pytest.fixture
    def file_paths(self):
        """Fixture that returns file paths for testing."""
        test_dir = os.path.dirname(__file__)
        return {
            'train': os.path.join(test_dir, "data/train.csv"),
            'test': os.path.join(test_dir, "data/test.csv"),
            'result': os.path.join(test_dir, "data/gender_submission.csv")
        }

    @pytest.fixture
    def titanic_instance(self, file_paths):
        """Fixture that creates and returns the required instances for testing."""
        preprocessing = TitanicPreprocessing(file_paths['train'], file_paths['test'])
        training = TitanicTraining()
        evaluation = TitanicEvaluation(file_paths['result'])
        
        # Create a wrapper object with all necessary methods
        class TitanicWrapper:
            def preprocess_data(self):
                return preprocessing.preprocess_data()
                
            def train_model(self, data):
                return training.train_model(data)
                
            def evaluate_model(self, model, data):
                return evaluation.evaluate_model(model, data)
                
        return TitanicWrapper()

    @pytest.fixture
    def preprocessed_data(self, titanic_instance):
        """Fixture that returns preprocessed training and test data."""
        return titanic_instance.preprocess_data()

    def test_get_preprocessed_data(self, preprocessed_data):
        """
        Test the get_preprocessed_data method of TitanicTraining class.
        """
        preprocessed_train_data, preprocessed_test_data = preprocessed_data
        assert preprocessed_train_data.shape == (891, 9)
        assert preprocessed_test_data.shape == (418, 10)

    def test_train_model(self, titanic_instance, preprocessed_data):
        """
        Test the train_model method of TitanicTraining class.
        """
        preprocessed_train_data, _ = preprocessed_data
        trained_model = titanic_instance.train_model(preprocessed_train_data)
        assert trained_model is not None

    def test_evaluate_model(self, titanic_instance, preprocessed_data):
        """
        Test the evaluate_model method of TitanicTraining class.
        """
        preprocessed_train_data, preprocessed_test_data = preprocessed_data
        trained_model = titanic_instance.train_model(preprocessed_train_data)
        evaluation_result = titanic_instance.evaluate_model(
            trained_model, preprocessed_test_data
        )
        assert evaluation_result is not None