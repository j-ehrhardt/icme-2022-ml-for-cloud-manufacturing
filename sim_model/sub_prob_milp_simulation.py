from sim_model.importexport import import_json, export_json
from random import getrandbits
from datetime import datetime
import pandas as pd
import pulp as pl

# Definition of size of sub problems to be optimized
sub_logistics = 1
sub_products = 5
sub_locations = 3

# Create empty dict to store all objective function values
obj_func_vals = dict()

# Data set naming
data_sets = ['truth',
             'lr',
             'ar',
             'lstm']
scenarios = ['small',
             'intermediate_small',
             'medium']

# Importing truth values
truth_data = {'small': import_json('data/truth/small_problem_data.json'),
              'intermediate_' + \
              'small': import_json('data/truth/intermediate_small_problem_data.json'),
              'medium': import_json('data/truth/medium_problem_data.json')}

# Importing lr predictions
lr_data_pre = {'small': {'lc': import_json('data/linreg_autoreg_predictions/predicted_linreg_log_values_small.json'),
                         'oc': import_json('data/linreg_autoreg_predictions/predicted_linreg_op_values_small.json')},
               'intermediate_' + \
               'small': {'lc': import_json('data/linreg_autoreg_predictions/' + \
                                           'predicted_linreg_log_values_intermediate_small.json'),
                         'oc': import_json('data/linreg_autoreg_predictions/' + \
                                           'predicted_linreg_op_values_intermediate_small.json')},
               'medium': {'lc': import_json('data/linreg_autoreg_predictions/predicted_linreg_log_values_medium.json'),
                         'oc': import_json('data/linreg_autoreg_predictions/predicted_linreg_op_values_medium.json')}}

# Simplification of lr dict structure
lr_data = dict()
for scenario in scenarios:
    lr_data[scenario] = dict()
    for cost_type in ['lc', 'oc']:
        lr_data[scenario][cost_type] = dict()
        for number in lr_data_pre[scenario][cost_type].keys():
            lr_data[scenario][cost_type][lr_data_pre[scenario][cost_type][number]['Name:']] = \
                lr_data_pre[scenario][cost_type][number]['predicted_values']

# Importing ar predictions
ar_data_pre = {'small': {'lc': import_json('data/linreg_autoreg_predictions/predicted_autoreg_log_values_small.json'),
                         'oc': import_json('data/linreg_autoreg_predictions/predicted_autoreg_op_values_small.json')},
               'intermediate_' + \
               'small': {'lc': import_json('data/linreg_autoreg_predictions/' + \
                                           'predicted_autoreg_log_values_intermediate_small.json'),
                         'oc': import_json('data/linreg_autoreg_predictions/' + \
                                           'predicted_autoreg_op_values_intermediate_small.json')},
               'medium': {'lc': import_json('data/linreg_autoreg_predictions/predicted_autoreg_log_values_medium.json'),
                         'oc': import_json('data/linreg_autoreg_predictions/predicted_autoreg_op_values_medium.json')}}

# Simplification of ar dict structure
ar_data = dict()
for scenario in scenarios:
    ar_data[scenario] = dict()
    for cost_type in ['lc', 'oc']:
        ar_data[scenario][cost_type] = dict()
        for number in ar_data_pre[scenario][cost_type].keys():
            ar_data[scenario][cost_type][ar_data_pre[scenario][cost_type][number]['Name:']] = \
                ar_data_pre[scenario][cost_type][number]['predicted_values']



# Importing ar predictions
ar_data_pre = dict()

# Importing ml predictions
def df_preprocessing(dataframe):  # Preprocessing for faulty csv files, now unused
    rows = dict()
    for counter, row in enumerate(dataframe.iterrows()):
        values_list = row[1][0].split(' ')
        values_list = list(float(value) for value in values_list)
        rows[counter] = values_list
    return pd.DataFrame(rows).transpose()

# Data location and import
read_ml_data = {'small': pd.read_csv('data/ml_predictions/res_exp_small_pred.csv', index_col=[0]),
                'intermediate_' + \
                'small': pd.read_csv('data/ml_predictions/res_exp_medium_pred.csv', index_col=[0]),
                'medium': pd.read_csv('data/ml_predictions/res_exp_large_pred.csv', index_col=[0])}

# Storing data sets in new dict and adapt index of data frames to end with period 527 (last third)
ml_data = dict()
for data_set in read_ml_data.keys():
    ml_data[data_set] = read_ml_data[data_set].set_index(pd.Index(range(528 - len(read_ml_data[data_set]), 528)))

# Calculate number of periods for which all data is available, some ml prediction data sets are slightly smaller
periods_to_calculate = list((i, 'Period ' + str(i + 1))
                            for i in range(528 - min(len(ml_data[scenario]) for scenario in scenarios), 528))

# Composing period data per scenario
scenario_period_data = dict()
for scenario in scenarios:
    scenario_period_data[scenario] = dict()
    for period in periods_to_calculate:
        scenario_period_data[scenario][period[1]] = dict()
        for cost_type in ['period_OC', 'period_LC']:
            scenario_period_data[scenario][period[1]][cost_type] = dict()
            if cost_type == 'period_OC':  # Adding operational cost first
                for operation in truth_data[scenario]['period_data'][period[1]][cost_type].keys():
                    scenario_period_data[scenario][period[1]][cost_type][operation] = dict()
                    for location in truth_data[scenario]['period_data'][period[1]][cost_type][operation].keys():
                        scenario_period_data[scenario][period[1]][cost_type][operation][location] = dict()

                        # Adding truth price to dict of prices
                        scenario_period_data[scenario][period[1]][cost_type][operation][location]['truth'] = \
                            truth_data[scenario]['period_data'][period[1]][cost_type][operation][location]

                        # Calculate position lr data set to match truth data periods
                        len_org = len(lr_data[scenario]['oc'][str(operation + '_' + location).replace(' ', '')])
                        len_per = len(periods_to_calculate)
                        len_dif = len_org - len_per
                        first_per = periods_to_calculate[0][0]
                        per = period[0]
                        locator = per - first_per + len_dif
                        # Adding lr predictions to dict of prices
                        scenario_period_data[scenario][period[1]][cost_type][operation][location]['lr'] = \
                            float(lr_data[scenario]['oc'][str(operation + '_' + location).replace(' ', '')]
                                  ['%s' % locator])

                        # Adding ar predictions to dict of prices, same period counter as lr data
                        scenario_period_data[scenario][period[1]][cost_type][operation][location]['ar'] = \
                            float(ar_data[scenario]['oc'][str(operation + '_' + location).replace(' ', '')]
                                  ['%s' % locator])

                        # Adding ml predictions to dict of prices
                        scenario_period_data[scenario][period[1]][cost_type][operation][location]['lstm'] = \
                            float(ml_data[scenario][str(operation + '_' + location).replace(' ', '')][period[0]])

            elif cost_type == 'period_LC':  # Adding logistics cost
                for logistic in truth_data[scenario]['period_data'][period[1]][cost_type].keys():
                    scenario_period_data[scenario][period[1]][cost_type][logistic] = dict()
                    for start_location in truth_data[scenario]['period_data'][period[1]][cost_type][logistic].keys():
                        scenario_period_data[scenario][period[1]][cost_type][logistic][start_location] = dict()
                        for finish_location in truth_data[scenario]['period_data'][period[1]] \
                                [cost_type][logistic].keys():
                            scenario_period_data[scenario][period[1]][cost_type][logistic] \
                                [start_location][finish_location] = dict()

                            # Adding truth price to dict of prices, same period counter as oc data
                            scenario_period_data[scenario][period[1]][cost_type][logistic] \
                                [start_location][finish_location]['truth'] = \
                                truth_data[scenario]['period_data'][period[1]][cost_type][logistic] \
                                    [start_location][finish_location]

                            # Adding lr predictions to dict of prices,
                            scenario_period_data[scenario][period[1]][cost_type][logistic] \
                                [start_location][finish_location]['lr'] = \
                                    float(lr_data[scenario]['lc'][str(logistic + '_' +
                                                                      start_location + '_' +
                                                                      finish_location).replace(' ', '')] \
                                              ['%s' % locator])

                            # Adding ar predictions to dict of prices, same period counter as lr data
                            scenario_period_data[scenario][period[1]][cost_type][logistic] \
                                [start_location][finish_location]['ar'] = \
                                    float(ar_data[scenario]['lc'][str(logistic + '_' +
                                                                      start_location + '_' +
                                                                      finish_location).replace(' ', '')] \
                                              ['%s' % locator])

                            # Adding ml predictions to dict of prices
                            scenario_period_data[scenario][period[1]][cost_type][logistic] \
                                [start_location][finish_location]['lstm'] = \
                                    float(ml_data[scenario][str(str(logistic + '_' +
                                                                      start_location + '_' +
                                                                      finish_location).replace(' ', ''))][period[0]])

# Iterating over all scenarios
for scenario in scenarios:
    obj_func_vals[scenario] = dict()
        
    # Reduction of problem size
    prob_logistics = \
        truth_data[scenario]['scenario_logistics'][len(truth_data[scenario]['scenario_logistics']) - sub_logistics:]
    prob_products = \
        truth_data[scenario]['scenario_products'][len(truth_data[scenario]['scenario_products']) - sub_products:]
    prob_locations = \
        truth_data[scenario]['scenario_locations'][len(truth_data[scenario]['scenario_locations']) - sub_locations:]
    
    # Copying variables
    prob_U = truth_data[scenario]['permanent_U']
    prob_operations_for_product = truth_data[scenario]['operations_for_product']
    
    # Deduction of operations present in sub problem
    prob_operations = list()
    for product in prob_products:
        for operation in truth_data[scenario]['operations_for_product'][product]:
            if not operation in prob_operations:
                prob_operations.append(operation)
    
    # Recalculating competencies to secure every operation is manufacturable in sub problem
    prob_C = dict()
    for operation in prob_operations:
        prob_C[operation] = dict()
        for location in prob_locations:
            prob_C[operation][location] = 0
    
    while not all((sum(prob_C[operation][location] for location in prob_locations) >= 1)
                  for operation in prob_operations):
        for operation in prob_operations:
            for location in prob_locations:
                prob_C[operation][location] = getrandbits(1)
    
    # Definition of subsequent operations per product and operation for problem
    prob_operations_subsequent = dict()
    for product in prob_products:
        for counter, operation in enumerate(prob_operations_for_product[product]):
            if counter + 1 < len(prob_operations_for_product[product]):
                next_operation = prob_operations_for_product[product][counter + 1]
                if not product in prob_operations_subsequent.keys():
                    prob_operations_subsequent[product] = list()
                prob_operations_subsequent[product].append((operation, next_operation))
    
    # Definition of finish locations per start location for scenario
    prob_routes = dict()
    for start_location in prob_locations:
        prob_routes[start_location] = list()
        for finish_location in prob_locations:
            if start_location is not finish_location:
                prob_routes[start_location].append(finish_location)
    
    # Combining period data
    '''
    prob_period_data = dict()
    for counter, period in enumerate(truth_data[scenario]['period_data'].keys()):
        prob_period_data[period] = {'period_OC': {}, 'period_LC': {}}
        for cost_type in prob_period_data[period].keys():
            if cost_type == 'period_OC':
                for product in prob_products:
                    for operation in prob_operations_for_product[product]:
                        prob_period_data[period][cost_type][operation] = dict()
                        for location in prob_locations:
                            col_name = operation.replace(' ', '') + '_' + \
                                       product.replace(' ', '') + '_' + \
                                       location.replace(' ', '')
                            prob_period_data[period][cost_type][operation][location] = \
                                imported_data_set[col_name][counter]
            elif cost_type == 'period_LC':
                for logistic_service in prob_logistics:
                    prob_period_data[period][cost_type][logistic_service] = dict()
                    for start_location in prob_locations:
                        prob_period_data[period][cost_type][logistic_service][start_location] = dict()
                        for finish_location in prob_locations:
                            col_name = logistic_service.replace(' ', '') + '_' + \
                                       start_location.replace(' ', '') + '_' + \
                                       finish_location.replace(' ', '')
                            prob_period_data[period][cost_type][logistic_service][start_location][finish_location] = \
                                imported_data_set[col_name][counter]
    '''

    # Iterating over all data sets
    for data_set in data_sets:
        obj_func_vals[scenario][data_set] = dict()
        # Iterating through all periods to be calculated
        for period in list(period[1] for period in periods_to_calculate):
            '''
            # Optimization model
            # ==================
            '''
            # Initialize problem and declare problem type
            time = datetime.now().strftime("%d.%m.%y_%H-%M-%S")
            lp_problem = pl.LpProblem('ICME22_%s_%s_%s' % (time, scenario, data_set), pl.LpMinimize)

            # Decision variable gamma
            print('%s: Adding decision variables gamma ...' % period)
            lp_gamma = dict()
            for product in prob_products:
                for operation in prob_operations_for_product[product]:
                    if operation not in lp_gamma.keys():
                        lp_gamma[operation] = dict()
                    lp_gamma[operation][product] = dict()
                    for location in prob_locations:
                        lp_gamma[operation][product][location] = pl.LpVariable('gamma Decision for %s of %s at %s'
                                                                               % (operation, product, location),
                                                                               cat='Binary')

            # Decision variable delta
            print('%s: Adding decision variables delta ...' % period)
            lp_delta = dict()
            for product in prob_products:
                if product in prob_operations_subsequent.keys():
                    for operations in prob_operations_subsequent[product]:
                        if not product in lp_delta.keys():
                            lp_delta[product] = dict()
                        lp_delta[product][tuple(operations)] = dict()
                        for service_provider in prob_logistics:
                            lp_delta[product][tuple(operations)][service_provider] = dict()
                            for start_location in prob_locations:
                                for finish_location in prob_routes[start_location]:
                                    lp_delta[product][tuple(operations)][service_provider] \
                                        [(start_location, finish_location)] \
                                        = pl.LpVariable('delta Decision for transport of %s from %s at %s to %s at %s by %s'
                                                        % (product, tuple(operations)[0], start_location,
                                                           operations[1], finish_location, service_provider),
                                                        cat='Binary')

            # Objective function
            print('%s: Adding objective function to the model ...' % period)
            lp_problem += pl.lpSum(scenario_period_data[scenario][period]['period_OC'][operation][location][data_set]
                                   * lp_gamma[operation][product][location]
                                   for location in prob_locations
                                   for product in prob_products
                                   for operation in prob_operations_for_product[product]) \
                          + pl.lpSum(scenario_period_data[scenario][period]['period_LC'] \
                                         [service_provider][start_location][finish_location][data_set]
                                     * lp_delta[product][tuple(operations)][service_provider] \
                                         [(start_location, finish_location)]
                                     for product in prob_operations_subsequent.keys()
                                     for operations in prob_operations_subsequent[product]
                                     for service_provider in prob_logistics
                                     for start_location in prob_locations
                                     for finish_location in prob_routes[start_location]), \
                          "Total cost of all operations and logistics"

            # Constraint 1
            print('%s: Adding group of constraints 1 to the model ...' % period)
            for product in prob_products:
                for operation in prob_operations_for_product[product]:
                    lp_problem += pl.lpSum(lp_gamma[operation][product][location] for location in prob_locations) \
                                  == prob_U[operation][product]

            # Constraint 2
            print('%s: Adding group of constraints 2 to the model ...' % period)
            for product in prob_operations_subsequent.keys():
                for operations in prob_operations_subsequent[product]:
                    for start_location in prob_locations:
                        for finish_location in prob_routes[start_location]:
                            lp_problem += lp_gamma[tuple(operations)[0]][product][start_location] \
                                          + lp_gamma[tuple(operations)[1]][product][finish_location] \
                                          - 1 \
                                          <= pl.lpSum(lp_delta[product][tuple(operations)]
                                                      [service_provider][(start_location, finish_location)]
                                                      for service_provider in prob_logistics)

            # Constraint 3
            print('%s: Adding group of constraints 3 to the model ...' % period)
            for product in prob_products:
                for operation in prob_operations_for_product[product]:
                    for location in prob_locations:
                        lp_problem += lp_gamma[operation][product][location] <= prob_C[operation][location]

            '''
            # Solving the model
            # =================
            '''
            print('%s: Solving the model ...' % period)
            # Call cbc for solving the problem
            path_to_cbc = '/opt/homebrew/bin/cbc'
            solver = pl.COIN_CMD(path=path_to_cbc, keepFiles=False)
            lp_problem.solve(solver)

            # Add objective function value to dict
            obj_func_vals[scenario][data_set][period] = lp_problem.objective.value()

# Export objective function values to json
export_json(obj_func_vals, 'data/objective_function_values.json')
