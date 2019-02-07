import os

datasets = ['brown', 'synthetic', 'yahoo', 'yelp', 'omniglot']

for dataset in datasets:
    dir_name = 'datasets/{}_data'.format(dataset)
    files = os.listdir(dir_name)

    for file in files:
        if file.endswith("t"):
            os.remove(os.path.join(dir_name, file))