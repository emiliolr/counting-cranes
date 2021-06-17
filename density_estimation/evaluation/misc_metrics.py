import numpy as np

def mean_percent_error(truth, pred):
    truth, pred = np.array(truth), np.array(pred)

    return np.mean(np.absolute((truth - pred) / truth))

#TODO: add GAME or patch metrics?

if __name__ == '__main__':
    #TESTING mean_percent_error:
    truth = [1, 2, 3]
    pred = [1, 3, 4]

    print(mean_percent_error(truth, pred))
