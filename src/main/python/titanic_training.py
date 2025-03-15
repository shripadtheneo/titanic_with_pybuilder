"""
A class that handles model training for the Titanic dataset.
"""

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
    A class that handles model training for the Titanic dataset.

    Methods:
    - train_model(preprocessed_train_data): Trains models using the
      preprocessed training data.
    """

    def train_model(self, preprocessed_train_data):
        """
        Trains models using the preprocessed training data.

        Parameters:
        - preprocessed_train_data (DataFrame): The preprocessed training data.

        Returns:
        - trained_model (dict): A dictionary containing the trained models.
        """
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
