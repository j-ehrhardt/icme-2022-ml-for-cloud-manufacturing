import pulp as pl
from datetime import datetime
from sim_model.importexport import import_json

'''
# Import problem data
'''
file = './sim_model/meta_inf/problem_data/small_problem_data.json'
imported_data = import_json(file)
operations_for_product = imported_data['operations_for_product']
permanent_U = imported_data['permanent_U']
permanent_C = imported_data['permanent_C']

'''
# Iterating through all periods
'''
for period in imported_data['period_data'].keys():
    break

    # Setting period dependent variables based on imported data
    period_data = imported_data['period_data'][period]

    period_locations = period_data['period_locations']
    period_logistics = period_data['period_logistics']
    period_products = period_data['period_products']
    period_routes = period_data['period_routes']
    period_operations_subsequent = period_data['period_operations_subsequent']
    period_OC = period_data['period_OC']
    period_LC = period_data['period_LC']

    '''
    # Optimization model
    # ==================
    '''
    # Initialize problem and declare problem type
    period_time = datetime.now().strftime("%d.%m.%y_%H-%M-%S")
    lp_problem = pl.LpProblem('ICME22_%s_%s' % (period, period_time), pl.LpMinimize)

    # Decision variable gamma
    print('%s: Adding decision variables gamma ...' % period)
    lp_gamma = dict()
    for product in period_products:
        for operation in operations_for_product[product]:
            if operation not in lp_gamma.keys():
                lp_gamma[operation] = dict()
            lp_gamma[operation][product] = dict()
            for location in period_locations:
                lp_gamma[operation][product][location] = pl.LpVariable('gamma Decision for %s of %s at %s'
                                                                       % (operation, product, location),
                                                                       cat='Binary')

    # Decision variable delta
    print('%s: Adding decision variables delta ...' % period)
    lp_delta = dict()
    for product in period_products:
        if product in period_operations_subsequent.keys():
            for operations in period_operations_subsequent[product]:
                    if not product in lp_delta.keys():
                        lp_delta[product] = dict()
                    lp_delta[product][tuple(operations)] = dict()
                    for service_provider in period_logistics:
                        lp_delta[product][tuple(operations)][service_provider] = dict()
                        for start_location in period_locations:
                            for finish_location in period_routes[start_location]:
                                lp_delta[product][tuple(operations)][service_provider][(start_location, finish_location)] \
                                    = pl.LpVariable('delta Decision for transport of %s from %s at %s to %s at %s by %s'
                                                    % (product, tuple(operations)[0], start_location,
                                                       operations[1], finish_location, service_provider),
                                                    cat='Binary')

    # Objective function
    print('%s: Adding objective function to the model ...' % period)
    lp_problem += pl.lpSum(period_OC[operation][product][location]
                            * lp_gamma[operation][product][location]
                            for location in period_locations
                            for product in period_products
                            for operation in operations_for_product[product]) \
                  + pl.lpSum(period_LC[service_provider][start_location][finish_location]
                             * lp_delta[product][tuple(operations)][service_provider][(start_location, finish_location)]
                             for product in period_operations_subsequent.keys()
                             for operations in period_operations_subsequent[product]
                             for service_provider in period_logistics
                             for start_location in period_locations
                             for finish_location in period_routes[start_location]), \
            "Total cost of all operations and logistics"

    # Constraint 1
    print('%s: Adding group of constraints 1 to the model ...' % period)
    for product in period_products:
        for operation in operations_for_product[product]:
            lp_problem += pl.lpSum(lp_gamma[operation][product][location] for location in period_locations) \
                          == permanent_U[operation][product]

    # Constraint 2
    print('%s: Adding group of constraints 2 to the model ...' % period)
    for product in period_operations_subsequent.keys():
        for operations in period_operations_subsequent[product]:
            for start_location in period_locations:
                for finish_location in period_routes[start_location]:
                    lp_problem += lp_gamma[tuple(operations)[0]][product][start_location] \
                                  + lp_gamma[tuple(operations)[1]][product][finish_location] \
                                  - 1 \
                                  <= pl.lpSum(lp_delta[product][tuple(operations)]
                                              [service_provider][(start_location, finish_location)]
                                              for service_provider in period_logistics)

    ''' Not using competency constraint temporarily'''
    print('%s: Adding group of constraints 3 to the model ...' % period)
    # Constraint 3
    for product in period_products:
        for operation in operations_for_product[product]:
            for location in period_locations:
                lp_problem += lp_gamma[operation][product][location] <= permanent_C[operation][location]

    '''
    # Solving the model
    # =================
    '''
    print('%s: Solving the model ...' % period)
    # Call cbc for solving the problem
    path_to_cbc = '/opt/homebrew/bin/cbc'
    solver = pl.COIN_CMD(path=path_to_cbc, keepFiles=True)
    lp_problem.solve(solver)
    #  lp_problem.writeMPS('model.mps')
    break
