from create_experiments import *
from data_module import *
from lstm_module import *
from eval_module import *
import multiprocessing as mp


def run(hparam: dict):
    """
    run a single experimental setup training run.
    setup is passed as dict.
    """
    # create_ds
    comp_X, comp_y, train_X, train_y, _, _,input_size, i = create_ds(hparam)
    # train model
    lstm = train_lstm(hparam, train_X, train_y, input_size)
    # eval and save results
    save_results(hparam, lstm, comp_X, comp_y, i)
    # visualize results
    # visualize_ml_predictions(hparam, lstm, comp_X, comp_y)
    print("Experiment " + str(hparam["EXPERIMENT_ID"]) + " finished...\n\n")
    return

def pretrained_eval_run(hparam: dict):
    """
    eval a trained model and evaluate it
    setup as passed as dict
    """
    comp_X, comp_y, train_X, train_y, test_X, test_y, input_size, i = create_ds(hparam)

    lstm = LSTM(hparam, input_size)  # we do not specify pretrained=True, i.e. do not load default weights
    lstm.load_state_dict(torch.load(hparam["CHECKPOINT_PATH"] + "/logs_exp" + str(hparam["EXPERIMENT_ID"]) + "/model.pth"))

    # eval and save results
    save_results(hparam, lstm, test_X, test_y, i)
    # visualize results
    visualize_ml_predictions(hparam, lstm, comp_X, comp_y)
    return

def full_run(hparam:dict, mode:str):
    """
    iterating through dict of dicts to run all experiments (e.g. in experiments.json)
    dict of dicts as input
    flag for "train" or "test"
    """
    if mode == "train":
        for key in hparam:
            run(hparam[key])
    if mode == "eval":
        for key in hparam:
            pretrained_eval_run(hparam[key])
            # eval on metrics
            df = results_to_df(hparam, n_smallest=5)
            find_best(df)
    return


if __name__ == "__main__":
    hparam = load_experiments()
    #hparam = create_exp()
    #launchTensorboard()
    full_run(hparam, mode="train")
    #for key in hparam:
    #    pretrained_eval_run(hparam[key])

