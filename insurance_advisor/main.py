from sklearn.metrics import classification_report, plot_confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

from insurance_advisor.advisor import InsuranceAdvisor


def train():
    features, prediction = InsuranceAdvisor.load_data('data/dataset_train.csv')

    advisor = InsuranceAdvisor()
    advisor.fit(features, prediction, algorithm='exact')
    # TODO find sweet spot for pseudocount so both train and test is good

    model_id = advisor.save_model()
    return model_id


def evaluate(model_id):
    features, prediction = InsuranceAdvisor.load_data('data/dataset_test.csv')

    advisor = InsuranceAdvisor()
    advisor.load_model(model_id)

    prediction_actual = advisor.predict(features)
    print(classification_report(prediction, prediction_actual))
    plot_confusion_matrix(advisor, features, prediction, values_format='d', cmap='Blues')
    plt.show()


def predict(model_id, filename):
    data, features = InsuranceAdvisor.load_data(filename)

    advisor = InsuranceAdvisor()
    advisor.load_model(model_id)

    predictions = advisor.predict(features)
    table = tabulate(pd.concat([data, predictions], axis=1), headers='keys', showindex=True)
    print(table)


if __name__ == '__main__':
    model_id = '2020-04-09T14-22-02'  # Accuracy: 0.82
    # model_id = train()
    evaluate(model_id)
    # predict(model_id, 'data/dataset_predict.csv')
