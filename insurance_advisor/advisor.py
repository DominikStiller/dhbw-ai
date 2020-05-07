import os
from datetime import datetime

import pandas as pd
from pomegranate import BayesianNetwork


class InsuranceAdvisor:
    _feature_cols = ['Geschlecht', 'Familienstand', 'Alter', 'Kinder',
                     'Bildungsstand', 'Beruf', 'Jahresgehalt', 'Immobilienbesitz']
    _prediction_col = 'Versicherungstarif'

    # Interface of scikit-learn classifier
    _estimator_type = 'classifier'
    classes_ = ['Tarif A', 'Tarif B', 'Tarif C', 'Tarif D', 'ablehnen']

    def __init__(self):
        self.model = None

    @staticmethod
    def load_data(filename, prediction=False):
        """Load and transform data from .csv file"""
        data = pd.read_csv(filename)
        data_original = data.copy()

        for feature in InsuranceAdvisor._feature_cols:
            assert feature in data.columns, "data must contain all feature columns"

        # Transform numerical data into categorical bins
        age_bin_edges = [-1, 20, 40, 60, 80, 100]
        age_bin_labels = ["0-20", "20-40", "40-60", "60-80", "80-100"]
        data['Alter'] = pd.cut(data['Alter'], age_bin_edges, labels=age_bin_labels)

        wage_bin_edges = [-1, 20e3, 40e3, 60e3, 80e3, 100e3, 120e3, 140e3]
        wage_bin_labels = ["0-20000", "20000-40000", "40000-60000", "60000-80000",
                           "80000-100000", "100000-120000", "120000-140000"]
        data['Jahresgehalt'] = pd.cut(data['Jahresgehalt'], wage_bin_edges, labels=wage_bin_labels)

        data['Kinder'] = data['Kinder'].apply(str)

        if not prediction:
            # Training / Testing
            return data[InsuranceAdvisor._feature_cols], data[InsuranceAdvisor._prediction_col]
        else:
            # Prediction
            return data_original[InsuranceAdvisor._feature_cols], data[InsuranceAdvisor._feature_cols]

    def fit(self, features, prediction, **kwargs):
        """Create a Bayesian network from the given samples"""
        data = pd.concat([features, prediction], axis='columns')

        self.model = BayesianNetwork.from_samples(X=data, state_names=data.columns, name="Insurance Advisor", **kwargs)
        self.model.freeze()

    def predict(self, features):
        """Get maximum likelihood estimate for each row"""
        return self.predict_probabilities(features).idxmax(axis='columns').rename('Vorhersage {}'.format(self._prediction_col))

    def predict_with_probability(self, features):
        """Get maximum likelihood estimate and probabilitz for each row"""
        probabilities = self.predict_probabilities(features)
        return probabilities.idxmax(axis='columns').rename('Prediction {}'.format(self._prediction_col)),\
               probabilities.max(axis='columns').rename('Probability')

    def predict_probabilities(self, features):
        """Get probabilities of each tarif for each row"""
        if not self.model:
            raise Exception("A model needs to be fitted or loaded before prediction")

        for feature in self._feature_cols:
            assert feature in features.columns, "data must contain all feature columns"

        features = features.copy()
        features.drop(self._prediction_col, axis=1, inplace=True, errors='ignore')
        features.insert(len(features.columns), self._prediction_col, None)

        # Predict probability distribution
        predicted_graph = self.model.predict_proba(features.values.tolist())
        # Get probability distribution for 'Versicherungstarif' column
        probabilities = [row[-1].parameters[0] for row in predicted_graph]
        # Convert to data frame and order classes
        return pd.DataFrame(probabilities)[self.classes_]

    def save_model(self):
        """Save a fitted model"""
        if not self.model:
            raise Exception("A model needs to be fitted or loaded before saving")

        model_id = datetime.now().replace(microsecond=0).isoformat().replace(":", "-")

        # Create directory and write to file
        os.makedirs('model', exist_ok=True)
        filename = 'model/advisor-{}.json'.format(model_id)
        with open(filename, 'w+') as file:
            file.write(self.model.to_json(indent=3))
        print("Model id:", model_id)

    def load_model(self, model_id):
        """Load a previously saved model"""
        with open('model/advisor-{}.json'.format(model_id), 'r') as file:
            self.model = BayesianNetwork.from_json(file.read())
