import json
import pandas as pd

def flatten_period(input):
    """
    flatteing dict structure into one-dim dict
    """
    row_dict = {}

    def flatten(x, name=""):
        if " " in name:
            name = name.replace(" ", "")
        if type(x) is dict:     # for dict type
            for a in x:
                flatten(x[a], name + a + "_")
        elif type(x) is list:   # for list type
            i = 0
            for a in x:
                flatten(a, name + str(i) + "_")
                i += 1
        else:
            row_dict[name[:-1]] = x
    flatten(input)
    return row_dict

def json_to_timeseries(json_path):
    """
    flattening a json file into a df-timeseries for processing with NN
    """
    with open(json_path) as file:
        sim = json.load(file)

    df = pd.DataFrame()
    ts_df = pd.DataFrame()

    for row in sim["period_data"]:
        oc_dict = flatten_period(sim["period_data"][row]["period_OC"])
        lc_dict = flatten_period(sim["period_data"][row]["period_LC"])
        oc_dict = oc_dict | lc_dict

        df = df.from_dict(oc_dict, orient="index")
        df = df.transpose()
        ts_df = pd.concat([ts_df, df], sort=False)

    ts_df = ts_df.reset_index(drop=True)
    return ts_df

def json_to_csv(json_path, csv_path):
    """
    writing df to csv
    """
    df = json_to_timeseries(json_path)
    df.to_csv(csv_path, index=False)
    return


if __name__ == "__main__":
    JSON_PATH = "../data/small_problem_data.json"
    CSV_PATH = "../data/small_problem_data.csv"

    json_to_csv(JSON_PATH, CSV_PATH)

