from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv('data/dataset_full.csv')

while True:
    train, test = train_test_split(data, test_size=.3, shuffle=True)
    # Try until train part contains all values from 0 to 7 for column 'Kinder'
    # Necessary because there are not many samples with high numbers of children
    if all([n in train['Kinder'].values for n in range(0, 8)]):
        break

train.to_csv('data/dataset_train.csv', index=False)
test.to_csv('data/dataset_test.csv', index=False)
test.drop('Versicherungstarif', axis='columns').head(n=10).to_csv('data/dataset_predict.csv', index=False)
