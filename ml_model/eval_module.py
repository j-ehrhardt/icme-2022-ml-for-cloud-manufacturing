import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from data_module import load_scaler
from create_experiments import *


def launchTensorboard(log_dir="logs/"):
    os.system('tensorboard --logdir=../ml_model/' + log_dir)
    return

def save_results(hparam: dict, model, ds_X, ds_y, i):
    """
    saving results and experimental setup as .json file in log-dir
    """
    rmse, mse, mae, rmse_oc, mse_oc, mae_oc, rmse_lc, mse_lc, mae_lc = eval_ml(hparam, model, ds_X, ds_y, i)

    results = {
        "setup":hparam,
        "results":{
        "RMSE":rmse,
        "MSE":mse,
        "MAE":mae,
        "RMSE_OC":rmse_oc,
        "MSE_OC":mse_oc,
        "MAE_OC":mae_oc,
        "RMSE_LC":rmse_lc,
        "MSE_LC":mse_lc,
        "MAE_LC":mae
    }}

    with open(hparam["CHECKPOINT_PATH"] + "logs_exp" + str(hparam["EXPERIMENT_ID"]) + "/results_exp" + str(hparam["EXPERIMENT_ID"])  + ".json", "w") as handle:
        json.dump(results, handle, indent=4)
        handle.close()

    print("Results for experiment: " + str(hparam["EXPERIMENT_ID"]) + "\n")
    for key in results:
        print(key + ": " + str(results[key]) + "\n")
    return

def vstack_arr(arr, i):
    arr1, arr2 = arr[0][:i], arr[0][i:]
    for row in arr:
        arr1 = np.vstack((arr1, row[:i]))
    for row in arr:
        arr2 = np.vstack((arr2, row[i:]))
    return arr1, arr2

def calc_metrics(pred, gt):
    """
    calc metrics for model predictions and ground truth
    model predicts within this fct
    """
    y = torch.Tensor(gt)
    y_hat = torch.Tensor(pred)

    print("y shape is ", y.shape)
    print("y_hat shape is", y_hat.shape)
    # root-mean-squared-error
    y = y.T
    y_hat = y_hat.T
    z, rmse = 1, 0
    for x,i in enumerate(y):
        rmse = rmse + torch.sqrt(torch.sum(y[x] - y_hat[x])**2) / y.__len__()
        z += 1
    rmse = rmse / z

    # mean-squared error
    z, mse = 1, 0
    for x,i in enumerate(y):
        mse = mse + torch.sum(y[x] - y_hat[x])**2 / y.__len__()

        z += 1
    mse = mse / z

    # mean-absolute-error
    z, mae = 1, 0
    for x,i in enumerate(y):
        mae = mae + torch.sum(y[x] - y_hat[x]) / y.__len__()
        z += 1
    mae = mae / z
    return rmse.item(), mse.item(), mae.item()

def eval_ml(hparam: dict, model, ds_X, ds_y, i):
    """
    eval different losses for oc, lc, and combined
    """
    scaler = load_scaler(hparam["EXPERIMENT_ID"], hparam["CHECKPOINT_PATH"])

    model.eval()
    predict_X = model(ds_X)

    # predict
    pred = predict_X.data.numpy()
    ground_truth = ds_y.data.numpy()

    # inverse scaler
    pred = scaler.inverse_transform(pred)
    ground_truth = scaler.inverse_transform(ground_truth)

    # calc combined metrics
    rmse, mse, mae = calc_metrics(pred, ground_truth)

    # calc split oc and lc metrics
    pred_oc, pred_lc = vstack_arr(pred, i)
    ground_truth_oc, ground_truth_lc = vstack_arr(ground_truth, i)

    # calc oc and lc specific
    rmse_oc, mse_oc, mae_oc = calc_metrics(pred_oc, ground_truth_oc)
    rmse_lc, mse_lc, mae_lc = calc_metrics(pred_lc, ground_truth_lc)

    return rmse, mse, mae, rmse_oc, mse_oc, mae_oc, rmse_lc, mse_lc, mae_lc

def mod_pred(hparam: dict, model, ds_X, ds_y):
    """
    model prediction
    """
    scaler = load_scaler(hparam["EXPERIMENT_ID"], hparam["CHECKPOINT_PATH"])

    model.eval()
    prediction = model(ds_X)

    # predict
    pred = prediction.data.numpy()
    ground_truth = ds_y.data.numpy()

    # inverse scaler
    pred = scaler.inverse_transform(pred)
    ground_truth = scaler.inverse_transform(ground_truth)
    return pred, ground_truth

def visualize_ml_predictions(hparam: dict, model, ds_X, ds_y):
    """
    plotting stuff
    """
    pred, ground_truth = mod_pred(hparam, model, ds_X, ds_y)

    # select channel
    i = pred[0][:5]
    j = ground_truth[0][:5]
    for row in pred: i = np.vstack((i, row[:5]))
    for row in ground_truth: j = np.vstack((j, row[:5]))

    # plotting
    p = int(len(ds_X)*0.7)
    plt.plot(i, label="ground_truth")
    plt.plot(j, label="prediction")
    plt.suptitle("Prediction results of five OC by our model")
    plt.axvline(x=p, c="orange", linestyle="--")
    plt.xlabel("periods")
    plt.ylabel("operating cost")
    plt.legend(loc="lower left")
    plt.show()
    return

def results_to_df(hparam:dict):
    """
    transforms result jsons from all logs_expX dirs into one df contianing all results
    """
    results = pd.DataFrame()
    # combining all individual result dicts into results dict
    for elem in os.listdir(hparam["CHECKPOINT_PATH"]):
        if "." not in elem: # check to be dir
            path = hparam["CHECKPOINT_PATH"] + str(elem)
            with open(path + "/results_" + elem[-4:] + ".json") as handle:
                res_dict = json.load(handle)
                res_dict = res_dict["results"]

                df = pd.DataFrame().from_dict(res_dict, orient="index")
                results[elem[-4:]] = df
    print(results)
    print("\n\n")
    return results

def find_best(df: pd.DataFrame, n_smallest):
    """
    finding the n smallest experimental setups in df regarding the different metrics
    """
    df = df.transpose()
    par = ["", "_OC", "_LC"]
    metrics = ["RMSE", "MSE", "MAE"]
    for m in metrics:
        print("\n\n" + m)
        for p in par:
            buff = df.nsmallest(n_smallest, m + p)
            print(p + ":")
            print(buff.index)
    return

if __name__ == "__main__":
    hparam = load_experiments()
    hparam = hparam["0"]

    #launchTensorboard(log_dir=hparam["CHECKPOINT_PATH"])
    df = results_to_df(hparam)
    find_best(df)

