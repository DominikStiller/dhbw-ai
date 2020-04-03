import json

with open('model/advisor-2020-04-02T20-28-05.json', 'r') as f:
    nodes = json.load(f)['states']

for node in nodes:
    chain = [node['name']]
    node = node['distribution']
    while 'parents' in node.keys():
        if len(node['parents']) > 1:
            print('XXX')
        node = node['parents'][0]
        if node['name'] == 'DiscreteDistribution':
            chain.append(",".join(node['parameters'][0].keys()))
        else:
            chain.append('X')

    print(" -> ".join(reversed(chain)))