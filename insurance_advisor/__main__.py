import sys

from sklearn.metrics import classification_report, plot_confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
from tabulate import tabulate

from advisor import InsuranceAdvisor


def train(train_data):
    features, prediction = InsuranceAdvisor.load_data(train_data)

    advisor = InsuranceAdvisor()
    advisor.fit(features, prediction, algorithm='exact')
    advisor.save_model()


def evaluate(model_id, evaluate_data):
    features, prediction = InsuranceAdvisor.load_data(evaluate_data)

    advisor = InsuranceAdvisor()
    advisor.load_model(model_id)

    prediction_actual = advisor.predict(features)
    print(classification_report(prediction, prediction_actual))

    plot_confusion_matrix(advisor, features, prediction, values_format='d', cmap='Blues')
    plt.savefig('confusion_matrix.png')
    print('Confusion matrix saved to "confusion_matrix.png"')
    plt.show()


def predict(model_id, predict_data):
    data, features = InsuranceAdvisor.load_data(predict_data, prediction=True)

    advisor = InsuranceAdvisor()
    advisor.load_model(model_id)

    predictions = advisor.predict_with_probability(features)
    table = tabulate(pd.concat([data, *predictions], axis=1), headers='keys', showindex=True)
    print("Predictions for", predict_data)
    print(table)


if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print('Usage: python insurance_advisor [command]')
        print('Available commands:')
        print('- train [train-data]')
        print('- evaluate [model-id] [evaluate-data]')
        print('- predict [model-id] [predict-data]')
        exit(1)

    command = sys.argv[1]
    if command == 'train':
        train(sys.argv[2])
    elif command == 'evaluate':
        evaluate(*sys.argv[2:4])
    elif command == 'predict':
        predict(*sys.argv[2:4])
    else:
        print('Unknown command')