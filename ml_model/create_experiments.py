import json

def make_exp_grid():
    """
    creates a grid of experiments from hyperparameters
    returns dict of dicts
    """

    seq_len = [4, 8, 16]
    learning_rate = [2e-3, 2e-4, 2e-5]
    max_epochs = [10, 50, 100, 200, 400, 800, 1600, 3200]
    experiments_dict = {}

    l = 0
    for m in seq_len:
        for i in learning_rate:
            for k in max_epochs:
                experiments_dict[l] = {
                    "EXPERIMENT_ID": l,
                    "DATA_PATH": "../data/small_problem_data.csv",
                    "CHECKPOINT_PATH": "logs/",
                    "SEQ_LEN": m,
                    "LEARNING_RATE": i,
                    "NUM_EPOCHS": k,
                    "NUM_LAYERS": 1,
                    "XZ_RATIO": 2,
                    "MAN_SEED": 42}
                l += 1
    return experiments_dict

def experiments_to_json(experiment_dict: dict):
    """
    dumps experimental grid in .json files
    """
    experiments = experiment_dict
    with open("experiments.json", "w") as json_file:
        json.dump(experiments, json_file, indent=4)

    print("experimental grid was created and saved in eperiments.json\n\n")
    return

def load_experiments():
    with open('experiments.json') as json_file:
        hparam = json.load(json_file)

    return hparam

def create_exp():
    experiments = make_exp_grid()
    experiments_to_json(experiments)
    hparam = load_experiments()
    return hparam

# Quickrun
if __name__ == "__main__":
    hparam = create_exp()

