from sim_model.importexport import import_json
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd

# Import objective function values for all scenarios and data sets from json file
imported_obj_func_vals = import_json('data/objective_function_values.json')

# Create dict of dataframes with scenarios as keys
obj_func_vals = dict()
for scenario in imported_obj_func_vals.keys():
    obj_func_vals[scenario] = pd.DataFrame(imported_obj_func_vals[scenario])

# Calculate errors for each dataframe
errors = dict()
for scenario in obj_func_vals.keys():
    for column in obj_func_vals[scenario]:
        if not column == 'truth':
            errors[scenario + '_' + column] = dict()
            print('Scenario: %s, Data set: %s' % (scenario, column))
            errors[scenario + '_' + column]['mae'] = mean_absolute_error(obj_func_vals[scenario]['truth'],
                                                      obj_func_vals[scenario][column])
            errors[scenario + '_' + column]['rmse'] = mean_squared_error(obj_func_vals[scenario]['truth'],
                                                      obj_func_vals[scenario][column], squared=False)
            errors[scenario + '_' + column]['mse'] = mean_squared_error(obj_func_vals[scenario]['truth'],
                                                     obj_func_vals[scenario][column])

errors = pd.DataFrame(errors)