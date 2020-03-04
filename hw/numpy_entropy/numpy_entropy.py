#!/usr/bin/env python3
import numpy as np

if __name__ == "__main__":

    # Load data distribution, each line containing a datapoint -- a string.
    n = 0
    distribution = dict()
    with open("numpy_entropy_data.txt", "r") as data:
        for line in data:
            line = line.rstrip("\n")
            n += 1
            if (not line in distribution):
              distribution[line] = {"data": 0, "model": 0}
            distribution[line]["data"] += 1

    for key in distribution:
        distribution[key]["data"] /= n

    # Load model distribution, each line `string \t probability`.
    with open("numpy_entropy_model.txt", "r") as model:
        for line in model:
            line = line.rstrip("\n")
            key, value = line.split("\t")
            if (key in distribution):
                distribution[key]["model"] = float(value)

    data_distribution = []
    model_distribution = []
    for item in distribution.values():
        data_distribution.append(item["data"])
        model_distribution.append(item["model"])

    np_data_distribution = np.array(data_distribution)
    np_model_distribution = np.array(model_distribution)

    # OUT: Compute and print the entropy H(data distribution)
    entropy = -(np_data_distribution * np.log(np_data_distribution)).sum()
    print("{:.2f}".format(entropy))

    # OUT: Compute and print cross-entropy H(data distribution, model distribution)
    cross_entropy = -(np_data_distribution * np.log(np_model_distribution)).sum()

    print("{:.2f}".format(cross_entropy))
    
    # OUT: KL-divergence D_KL(data distribution, model_distribution)
    print("{:.2f}".format(cross_entropy - entropy))
