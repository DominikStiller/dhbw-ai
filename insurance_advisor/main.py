from sklearn.metrics import accuracy_score, plot_confusion_matrix
import matplotlib.pyplot as plt

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


if __name__ == '__main__':
    # model_id = train()
    model_id = '2020-04-03T14-32-58'  # Accuracy: 0.78
    evaluate(model_id)
