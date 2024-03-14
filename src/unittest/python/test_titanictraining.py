from titanictraining import TitanicTraining
import os
import sys


class TestTitanicTraining:
    """
    This class contains unit tests for the TitanicTraining class.
    """

    test_dir = sys.path[0]
    train_data = os.path.join(test_dir, "data/train.csv")
    test_data = os.path.join(test_dir, "data/test.csv")
    test_result = os.path.join(test_dir, "data/gender_submission.csv")

    def test_get_preprocessed_data(self):
        """
        Test the get_preprocessed_data method of TitanicTraining class.
        """
        titanic = TitanicTraining(self.train_data, self.test_data, self.test_result)
        preprocessed_train_data, preprocessed_test_data = titanic.preprocess_data()
        assert preprocessed_train_data.shape == (891, 9)
        assert preprocessed_test_data.shape == (418, 10)

    def test_train_model(self):
        """
        Test the train_model method of TitanicTraining class.
        """
        titanic = TitanicTraining(self.train_data, self.test_data, self.test_result)
        preprocessed_train_data, preprocessed_test_data = titanic.preprocess_data()
        trained_model = titanic.train_model(preprocessed_train_data)
        assert trained_model is not None

    def test_evaluate_model(self):
        """
        Test the evaluate_model method of TitanicTraining class.
        """
        titanic = TitanicTraining(self.train_data, self.test_data, self.test_result)
        preprocessed_train_data, preprocessed_test_data = titanic.preprocess_data()
        trained_model = titanic.train_model(preprocessed_train_data)
        evaluation_result = titanic.evaluate_model(
            trained_model, preprocessed_test_data
        )
        assert evaluation_result is not None
