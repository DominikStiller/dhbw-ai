from sklearn.metrics import accuracy_score, plot_confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

from insurance_advisor.advisor import InsuranceAdvisor


def train():
    features, prediction = InsuranceAdvisor.load_data('data/dataset_train.csv')

    advisor = InsuranceAdvisor()
    advisor.fit(features, prediction, algorithm='exact', reduce_dataset=False)

    model_id = advisor.save_model()
    return model_id


def evaluate(model_id):
    features, prediction = InsuranceAdvisor.load_data('data/dataset_test.csv')

    advisor = InsuranceAdvisor()
    advisor.load_model(model_id)

    prediction_actual = advisor.predict(features)
    print("Accuracy:", accuracy_score(prediction, prediction_actual))
    plot_confusion_matrix(advisor, features, prediction, cmap='Blues')
    plt.show()


def predict(model_id, filename):
    data, features = InsuranceAdvisor.load_data(filename)

    advisor = InsuranceAdvisor()
    advisor.load_model(model_id)

    predictions = advisor.predict(features)
    full_table = [row.values for _, row in pd.concat([data, predictions], axis=1).iterrows()]
    print(tabulate(full_table, headers=(InsuranceAdvisor._feature_cols + [InsuranceAdvisor._prediction_col])))


if __name__ == '__main__':
    # model_id = train()
    model_id = '2020-04-03T14-32-58'  # Accuracy: 0.78
    # evaluate(model_id)
    predict(model_id, 'data/dataset_predict.csv')
