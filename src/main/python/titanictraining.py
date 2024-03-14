"""
A class that represents the training process for the Titanic dataset.
"""
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    BaggingClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


class TitanicTraining:
    """
    A class that represents the training process for the Titanic dataset.

    Parameters:
    - train_data_path (str): The file path to the training data.
    - test_data_path (str): The file path to the test data.
    - test_result_path (str): The file path to the test result.

    Methods:
    - preprocess_data(): Preprocesses the training and test data.
    - train_model(preprocessed_train_data): Trains a model using the
    preprocessed training data.
    - evaluate_model(trained_model, preprocessed_test_data): Evaluates the
    trained model using the preprocessed test data.
    - run(): Runs the entire training process and returns the evaluation
    results.
    """

    def __init__(self, train_data_path, test_data_path, test_result_path):
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.test_result_path = test_result_path

    def preprocess_data(self):
        """
        Preprocesses the training and test data.

        Returns:
        - preprocessed_train_data: The preprocessed training data.
        - preprocessed_test_data: The preprocessed test data.
        """
        # Add preprocessing logic here
        # ...
        train = pd.read_csv(self.train_data_path)
        test = pd.read_csv(self.test_data_path)
        test["Survived"] = ""

        train_test_data = [train, test]  # combine dataset

        for dataset in train_test_data:
            dataset["Title"] = dataset["Name"].str.extract(
                " ([A-Za-z]+)\.", expand=False
            )

        title_mapping = {
            "Mr": 0,
            "Miss": 1,
            "Mrs": 2,
            "Master": 3,
            "Dr": 3,
            "Rev": 3,
            "Col": 3,
            "Major": 3,
            "Mlle": 3,
            "Countess": 3,
            "Ms": 3,
            "Lady": 3,
            "Jonkheer": 3,
            "Don": 3,
            "Dona": 3,
            "Mme": 3,
            "Capt": 3,
            "Sir": 3,
        }

        for dataset in train_test_data:
            dataset["Title"] = dataset["Title"].map(title_mapping)

        train.drop("Name", axis=1, inplace=True)
        test.drop("Name", axis=1, inplace=True)

        sex_mapping = {"male": 0, "female": 1}
        for dataset in train_test_data:
            dataset["Sex"] = dataset["Sex"].map(sex_mapping)

        train["Age"].fillna(
            train.groupby("Title")["Age"].transform("median"), inplace=True
        )
        test["Age"].fillna(
            test.groupby("Title")["Age"].transform("median"), inplace=True
        )

        train.drop("Ticket", axis=1, inplace=True)
        test.drop("Ticket", axis=1, inplace=True)

        for dataset in train_test_data:
            dataset.loc[dataset["Age"] <= 16, "Age"] = 0
            dataset.loc[(dataset["Age"] > 16) & (dataset["Age"] <= 26),
                        "Age"] = 1
            dataset.loc[(dataset["Age"] > 26) & (dataset["Age"] <= 36),
                        "Age"] = 2
            dataset.loc[(dataset["Age"] > 36) & (dataset["Age"] <= 62),
                        "Age"] = 3
            dataset.loc[dataset["Age"] > 62, "Age"] = 4

        for dataset in train_test_data:
            dataset["Embarked"] = dataset["Embarked"].fillna("S")

        embarked_mapping = {"S": 0, "C": 1, "Q": 2}
        for dataset in train_test_data:
            dataset["Embarked"] = dataset["Embarked"].map(embarked_mapping)

        train["Fare"].fillna(
            train.groupby("Pclass")["Fare"].transform("median"), inplace=True
        )
        test["Fare"].fillna(
            test.groupby("Pclass")["Fare"].transform("median"), inplace=True
        )

        for dataset in train_test_data:
            dataset.loc[dataset["Fare"] <= 17, "Fare"] = 0
            dataset.loc[(dataset["Fare"] > 17) & (dataset["Fare"] <= 30),
                        "Fare"] = 1
            dataset.loc[(dataset["Fare"] > 30) & (dataset["Fare"] <= 100),
                        "Fare"] = 2
            dataset.loc[dataset["Fare"] > 100, "Fare"] = 3

        for dataset in train_test_data:
            dataset["Cabin"] = dataset["Cabin"].str[:1]

        cabin_mapping = {
            "A": 0,
            "B": 0.4,
            "C": 0.8,
            "D": 1.2,
            "E": 1.6,
            "F": 2,
            "G": 2.4,
            "T": 2.8,
        }
        for dataset in train_test_data:
            dataset["Cabin"] = dataset["Cabin"].map(cabin_mapping)

        train["Cabin"].fillna(
            train.groupby("Pclass")["Cabin"].transform("median"), inplace=True
        )
        test["Cabin"].fillna(
            test.groupby("Pclass")["Cabin"].transform("median"), inplace=True
        )

        train["FamilySize"] = train["SibSp"] + train["Parch"] + 1
        test["FamilySize"] = test["SibSp"] + test["Parch"] + 1

        family_mapping = {
            1: 0,
            2: 0.4,
            3: 0.8,
            4: 1.2,
            5: 1.6,
            6: 2,
            7: 2.4,
            8: 2.8,
            9: 3.2,
            10: 3.6,
            11: 4,
        }
        for dataset in train_test_data:
            dataset["FamilySize"] = dataset["FamilySize"].map(family_mapping)

        features_drop = ["SibSp", "Parch"]
        train = train.drop(features_drop, axis=1)
        test = test.drop(features_drop, axis=1)

        train = train.drop(["PassengerId"], axis=1)

        preprocessed_train_data = train
        preprocessed_test_data = test

        return preprocessed_train_data, preprocessed_test_data

    def train_model(self, preprocessed_train_data):
        """
        Trains a model using the preprocessed training data.

        Parameters:
        - preprocessed_train_data (DataFrame): The preprocessed training data.

        Returns:
        - trained_model (dict): A dictionary containing the trained models.

        """
        # Add model training logic here
        # ...
        x_train = preprocessed_train_data.drop("Survived", axis=1)
        y_train = preprocessed_train_data["Survived"]

        # KNN
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(x_train, y_train)

        # Decision Tree
        decision_tree = DecisionTreeClassifier()
        decision_tree.fit(x_train, y_train)

        # Random Forest
        random_forest = RandomForestClassifier(n_estimators=100)
        random_forest.fit(x_train, y_train)

        # Naive Bayes
        gaussian = GaussianNB()
        gaussian.fit(x_train, y_train)

        # SVM
        svc = SVC()
        svc.fit(x_train, y_train)

        # Extra Tree
        extra_tree = ExtraTreeClassifier()
        extra_tree.fit(x_train, y_train)

        # Extra Trees
        extra_trees = ExtraTreesClassifier()
        extra_trees.fit(x_train, y_train)

        # Bagging
        bagging = BaggingClassifier()
        bagging.fit(x_train, y_train)

        # AdaBoost
        ada_boost = AdaBoostClassifier()
        ada_boost.fit(x_train, y_train)

        # Gradient Boosting
        gradient_boosting = GradientBoostingClassifier()
        gradient_boosting.fit(x_train, y_train)

        trained_model = {
            "knn": knn,
            "decision_tree": decision_tree,
            "random_forest": random_forest,
            "gaussian": gaussian,
            "svc": svc,
            "extra_tree": extra_tree,
            "extra_trees": extra_trees,
            "bagging": bagging,
            "ada_boost": ada_boost,
            "gradient_boosting": gradient_boosting,
        }

        return trained_model

    def evaluate_model(self, trained_model, preprocessed_test_data):
        """
        Evaluates the trained model using the preprocessed test data.

        Parameters:
        trained_model (dict): A dictionary containing the trained models.
        preprocessed_test_data (DataFrame): The preprocessed test data.

        Returns:
        dict: A dictionary containing the evaluation results
        for each model.
        """
        # Add model evaluation logic here
        # ...
        x_test = preprocessed_test_data.drop(
            ["Survived", "PassengerId"], axis=1
        )
        test_result = pd.read_csv(self.test_result_path)
        y_test = test_result["Survived"]

        evaluation_results = {}
        for model_name, model in trained_model.items():
            evaluation_results[model_name] = model.score(x_test, y_test)

        return evaluation_results

    def run(self):
        """
        Runs the training process for the Titanic model.

        Returns:
            evaluation_results (dict): A dictionary containing the
            evaluation results of the trained model.
        """
        preprocessed_train_data, preprocessed_test_data = \
            self.preprocess_data()

        trained_model = self.train_model(preprocessed_train_data)
        evaluation_results = self.evaluate_model(
            trained_model, preprocessed_test_data
        )
        return evaluation_results
