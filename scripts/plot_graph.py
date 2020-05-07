from pomegranate import BayesianNetwork

# Requirements:
# - pygraphviz = "*" in Pipfile
# - Installation of https://www.graphviz.org/download/ (executable and dev package)

def plot_model(model_id):
    """Load a previously saved model"""
    with open('model/advisor-{}.json'.format(model_id), 'r') as file:
        model = BayesianNetwork.from_json(file.read())
    model.plot('graph.pdf')

plot_model('default')
