#!/usr/bin/env python

import pulp as pl
from math import sin, pi
from scipy.stats import truncnorm
from numpy.random import seed as np_seed
from numpy import mean
from math import sqrt
from random import seed as rand_seed
from random import getrandbits
from sim_model.italiancities import top100, distances
from sim_model.inflationdata import inflation
from sim_model.importexport import export_json

'''
Functions
'''
# Function for creating base lists
def baselist(f_what, f_length):
    return ["%s %s" % (f_what, i) for i in range(1, f_length + 1)]


# Function for generating list of all forward combinations of other list's elements
def all_combinations(f_in_list):
    f_out_list = list()
    for length in range(len(f_in_list)):
        for combination in pl.combination(f_in_list, length + 1):
            f_out_list.append(combination)
    return f_out_list


'''
Definition of scenarios, number of logistic service providers, operations, locations
'''
scenarios = {'small': {'number_logistics': 2,
                       'number_products': 10,
                       'number_locations': 5},
             'intermediate \
             small': {'number_logistics': 4,
                        'number_products': 50,
                        'number_locations': 10},
             'medium': {'number_logistics': 6,
                        'number_products': 100,
                        'number_locations': 15},
             'intermediate \
             large': {'number_logistics': 8,
                        'number_products': 250,
                        'number_locations': 25},
             'large': {'number_logistics': 10,  # max 10
                       'number_products': 500,  # max 1023
                       'number_locations': 50}}  # max 100

'''
Definition of base sets for all periods
'''
# List of locations [m]
all_locations = top100('./sim_model/meta_inf/it_municipalities.csv')[:100]

# List of logistic services [j]
all_logistics = baselist("Logistic Service", 10)

# List of operations [i]
all_operations = baselist("Operation", 10)

# List of products [k]
all_products = baselist("Product", len(all_combinations(all_operations)))

# Dictionary of operation subsets per product [i ∈ I(k)]
operations_for_product = dict()
for count, product in enumerate(all_products):
    operations_for_product[product] = all_combinations(all_operations)[count]

'''
Iterate over scenarios and generate data per scenario
'''
for scenario in scenarios.keys():
    print('Start building scenario %s' % scenario)
    # Initialize new dict to store scenario data
    scenario_data = dict()

    # Reset seed for random numbers
    seed = 42
    np_seed(seed)
    rand_seed(seed)

    # Definition of file name for data export
    export_file = '../%s_problem_data.json' % scenario

    # Inflation for each period (period length is 1/4 month ≈ 1 week, 48 periods per year)
    imported_inflation = inflation('./sim_model/meta_inf/it_inflation.csv')

    # Build list of periods based on inflation data set size
    periods = list('Period %s' % period for period in range(1, len(imported_inflation) +1 ))

    # ==================================================================================================================

    # Definition of competence parameter C, defining which operations can be executed at which locations
    permanent_C = dict()
    for operation in all_operations:
        permanent_C[operation] = dict()
        for location in all_locations:
            permanent_C[operation][location] = 0

    while not all((sum(permanent_C[operation][location] for location in all_locations) >= 1)
                  for operation in all_operations):
        for operation in all_operations:
            for location in all_locations:
                permanent_C[operation][location] = getrandbits(1)

    # Definition of period independent binary parameter U, defining which operations go into which product
    permanent_U = dict()
    for operation in all_operations:
        permanent_U[operation] = dict()
        for product in all_products:
            if operation in operations_for_product[product]:
                permanent_U[operation][product] = 1
            elif operation not in operations_for_product[product]:
                permanent_U[operation][product] = 0
            else:
                raise ValueError('Failed to build parameter U while looping through operations and products.')

    # Definition of period independent parameter d, by importing distances between all locations
    permanent_d = distances('./sim_model/meta_inf/results_distances.json')

    # ==================================================================================================================

    # Definition of subset of logistic services in scenario
    scenario_logistics = all_logistics[:scenarios[scenario]['number_logistics']]

    # Definition of subset of products to be produced in scenario
    scenario_products = all_products[:scenarios[scenario]['number_products']]

    # Definition of subset of operations required for manufacturing of products of scenario
    scenario_operations = list()
    for product in scenario_products:
        for operation in operations_for_product[product]:
            if not operation in scenario_operations:
                scenario_operations.append(operation)

    # Definition of subset of locations participating in period
    # Securing every operation being available at least once by potentially reassigning parameter C
    scenario_locations = all_locations[:scenarios[scenario]['number_locations']]
    while not all(sum(permanent_C[operation][location] >= 1 for location in scenario_locations)
                  for operation in all_operations):
        print('Reassigning competence parameter C for %s have all operations available at least once.' % scenario)
        for operation in all_operations:
            for location in all_locations:
                permanent_C[operation][location] = getrandbits(1)

    # Definition of finish locations per start location for scenario
    scenario_routes = dict()
    for start_location in scenario_locations:
        scenario_routes[start_location] = list()
        for finish_location in scenario_locations:
            if start_location is not finish_location:
                scenario_routes[start_location].append(finish_location)

    # Definition of subsequent operations per product and operation for scenario
    scenario_operations_subsequent = dict()
    for product in scenario_products:
        for counter, operation in enumerate(operations_for_product[product]):
            if counter + 1 < len(operations_for_product[product]):
                next_operation = operations_for_product[product][counter + 1]
                if not product in scenario_operations_subsequent.keys():
                    scenario_operations_subsequent[product] = list()
                scenario_operations_subsequent[product].append((operation, next_operation))

    # ==================================================================================================================

    '''Base value and factors for production time t'''
    # Production time per operation, drawn from truncated normal distribution
    lower = 3
    upper = 8
    mu = 4.5
    sigma = 2
    N = len(all_operations)
    samples = truncnorm.rvs((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma, size=N)
    base_t_operations = dict()
    for counter, operation in enumerate(scenario_operations):
        base_t_operations[operation] = samples[counter]

    # Time per operation factor for every location, drawn from truncated normal distribution
    lower = 0
    upper = 2
    mu = 1
    sigma = 0.1
    N = len(all_locations)
    samples = truncnorm.rvs((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma, size=N)
    base_t_locations = dict()
    for counter, location in enumerate(scenario_locations):
        base_t_locations[location] = samples[counter]

    # Time per operation seasonal factor for every period; cosinus function
    base_t_seasonal_impact = 0.1
    base_t_seasonal = dict()
    for counter, period in enumerate(periods):
        base_t_seasonal[period] = 1 - abs(sin((counter + 1) * pi / 48)) * base_t_seasonal_impact
        # 48 periods per year; 12 month * 4 periods

    # Productivity when executing operations
    base_productivity = dict()
    for counter, period in enumerate(periods):
        base_productivity[period] = dict()
        operation_executions = dict()
        for product in scenario_products:
            for operation in operations_for_product[product]:
                if operation not in operation_executions.keys():
                    operation_executions[operation] = counter
                elif operation in operation_executions.keys():
                    operation_executions[operation] += counter
        for operation in scenario_operations:
            base_productivity[period][operation] = 1 + sqrt(operation_executions[operation]) / 1000

    '''Base value and factors for production cost oc'''
    # Cost per hour per operation, drawn from truncated normal distribution
    lower = 50
    upper = 100
    mu = 70
    sigma = 10
    N = len(all_operations)
    samples = truncnorm.rvs((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma, size=N)
    base_oc_operations = dict()
    for counter, operation in enumerate(scenario_operations):
        base_oc_operations[operation] = samples[counter]

    # Cost per hour factor for every location, drawn from truncated normal distribution
    base_oc_locations = base_t_locations

    # Inflation factor per period
    base_oc_seasonal = dict()
    for counter, period in enumerate(periods):
        base_oc_seasonal[period] = imported_inflation[counter][0] / 100

    '''Base value and factors for logistics cost lc'''
    # Cost per kilometer for every logistics service provider, drawn from truncated normal distribution
    lower = 0.5
    upper = 1
    mu = 0.7
    sigma = 0.2
    N = len(all_logistics)
    samples = truncnorm.rvs((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma, size=N)
    base_lc_provider = dict()
    for counter, service_provider in enumerate(scenario_logistics):
        base_lc_provider[service_provider] = samples[counter]

    # Seasonal factor (inflation) for logistic service provider prices
    base_lc_seasonal = base_oc_seasonal

    # Competition factor for logistics service providers per period, drawn from truncated normal distribution
    lower = 0.01
    upper = 0.2
    mu = 0.10
    sigma = 0.04
    N = len(all_logistics)
    base_lc_competition = dict()
    for period in periods:
        base_lc_competition[period] = dict()
        samples = truncnorm.rvs((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma, size=N)
        for counter, service_provider in enumerate(scenario_logistics):
            base_lc_competition[period][service_provider] = 1 - samples[counter]

    # ==================================================================================================================

    '''
    # Iterating through all periods and calculating vector data
    '''
    period_data = dict()

    for period_counter, period in enumerate(periods):
        period_counter += 1

        period_data[period] = dict()
        print('Data generation for %s, %s' % (scenario, period))


        '''
        # Definition of parameters per period
        '''
        # Time of performing operation i in city m
        period_t = dict()
        for operation in scenario_operations:
            period_t[operation] = dict()
            for location in scenario_locations:
                period_t[operation][location] = base_t_operations[operation] \
                                                    * base_t_locations[location] \
                                                    * base_t_seasonal[period]

        # Cost of performing operation i in city m
        period_oc = dict()
        for operation in scenario_operations:
            period_oc[operation] = dict()
            for location in scenario_locations:
                period_oc[operation][location] = base_oc_operations[operation] \
                                                    * base_oc_locations[location] \
                                                    * base_oc_seasonal[period]

        # Cost of performing operation i in city m
        period_OC = dict()
        for operation in scenario_operations:
            period_OC[operation] = dict()
            for location in scenario_locations:
                period_OC[operation][location] = (period_oc[operation][location]
                                                           * period_t[operation][location]) \
                                                          / base_productivity[period][operation]

        # Cost of logistics service j
        period_lc = dict()
        for service_provider in scenario_logistics:
            period_lc[service_provider] = base_lc_provider[service_provider] \
                                          * base_lc_seasonal[period] \
                                          * base_lc_competition[period][service_provider]

        # Cost of logistic service j between city m and m'
        period_LC = dict()
        for service_provider in scenario_logistics:
            period_LC[service_provider] = dict()
            for start_location in scenario_locations:
                period_LC[service_provider][start_location] = dict()
                for finish_location in scenario_locations:
                    if start_location is finish_location:
                        period_LC[service_provider][start_location][finish_location] = 0
                    elif start_location is not finish_location:
                        period_LC[service_provider][start_location][finish_location] = \
                            period_lc[service_provider] * permanent_d[start_location][finish_location]

        for data in ('period_OC',
                     'period_LC'):
            loc = locals()
            period_data[period][data] = loc[data]

    for data in ('scenario_locations',
                 'scenario_logistics',
                 'scenario_operations',
                 'scenario_products',
                 'operations_for_product',
                 'permanent_C',
                 'permanent_U',
                 'period_data'):
        loc = locals()
        scenario_data[data] = loc[data]

    print('Writing json file scenario %s' % scenario)
    export_json(scenario_data, export_file)
