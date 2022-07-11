import torch
import pandas as pd
import numpy as np
import pickle
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from create_experiments import *

def fetch_ds(data_path):
    """
    fetch csv file
    """
    df = pd.read_csv(data_path, index_col=False)
    input_size = df.shape[1]
    i = 0
    for column in df:
        if "Logistic" in column:
            i += 1
    df = df.values
    return df, input_size, i

def sliding_window_ds(df, seq_len):
    """
    sampling csv ds into sliding window dataset of windowsize seq_len
    """
    X = []
    y = []

    for i in range(len(df)-seq_len-1):
        _X = df[i:(i+seq_len)]
        _y = df[i+seq_len]
        X.append(_X)
        y.append(_y)

    return np.array(X),np.array(y)

def save_scaler(scaler, id, saving_path):
    """
    saving scaler in logs
    """
    # save scaler as pickle in "./logs"
    with open(saving_path + "scaler_exp" + str(id) + ".pkl", "wb") as handle:
        pickle.dump(scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return

def load_scaler(id, saving_path):
    """
    loading scaler
    """
    scaler = pickle.load(open(saving_path + "scaler_exp" + str(id) + ".pkl", 'rb'))
    return scaler

def create_ds(hparam):
    """
    creating train and test ds
    """
    df, input_size, i = fetch_ds(hparam["DATA_PATH"])

    scaler = MinMaxScaler() # StandardScaler()
    df = scaler.fit_transform(df)
    save_scaler(scaler, hparam["EXPERIMENT_ID"], hparam["CHECKPOINT_PATH"])

    X, y = sliding_window_ds(df, hparam["SEQ_LEN"])

    # 70/30 train test split
    comp_X = Variable(torch.Tensor(np.array(X)))
    comp_y = Variable(torch.Tensor(np.array(y)))

    train_X = Variable(torch.Tensor(np.array(X[0:int(len(y) * 0.7)])))
    train_y = Variable(torch.Tensor(np.array(y[0:int(len(y) * 0.7)])))

    test_X = Variable(torch.Tensor(np.array(X[int(len(y)*0.7):])))
    test_y = Variable(torch.Tensor(np.array(y[int(len(y)*0.7):])))
    return comp_X, comp_y, train_X, train_y, test_X, test_y, input_size, i


if __name__ == "__main__":
    hparam = load_experiments()
    hparam = hparam["0"]

    a, b, c, d, e, f = create_ds(hparam=hparam)
    print(a, b, c, d, e, f)



