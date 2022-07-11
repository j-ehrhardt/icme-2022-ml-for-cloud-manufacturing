'''
This file is used to do geocoding for italian municipalities and to calculate distances to and from every location
to all other locations using the awesome openrouteservice pelias geocoding and matrix route calculation endpoints.
'''

import openrouteservice
import json
from time import sleep
from sim_model.italiancities import top100

# Retrive openrouteservice api key from website and store as .apikey file
with open('./sim_model/meta_inf/openrouteservice.apikey') as f:
    api_key = f.readline()

client = openrouteservice.Client(key=api_key)
country='ITA'  # Limit search to italy
locations = top100('./sim_model/meta_inf/it_municipalities.csv')

# Function implementing openrouteservice geocoding functionality
def geocoder(f_location):
    f_results = openrouteservice.geocode.pelias_search(text=f_location,
                                                       client=client,
                                                       country=country)
    return f_results


# Collect geocoding responses in dictionary
def collect_geocoding_results():
    f_results = dict()
    for location in locations:
        f_results[location] = geocoder(location)
    return f_results


# Import geocoding results from json file
def import_geocoding_results(results_json):
    with open(results_json) as json_file:
        return json.load(json_file)

# Parsing geocoding results for coordinates, coordinate format is (longitude, latitude)
location_coordinates = dict()
for location, data in import_geocoding_results('./sim_model/meta_inf/results_geocoding.json').items():
    location_coordinates[location] = data['features'][0]['geometry']['coordinates']

# Calculate distances for every combination of any two locations using openrouteservice matrix calculation endpoint
distances = dict()
for source_location in location_coordinates:
    distances[source_location] = dict()

    reduced_location_coordinates = dict()
    for destination_location in location_coordinates:
        if source_location is not destination_location:
            reduced_location_coordinates[destination_location] = location_coordinates[destination_location]

    source_coordinates = [location_coordinates[source_location]]
    destination_coordinates = list()
    for destination_location in reduced_location_coordinates:
        destination_coordinates.append(reduced_location_coordinates[destination_location])
    request_coordinates = source_coordinates + destination_coordinates
    print('Retriving routing data from "' + source_location + '" to all other locations')
    distance_matrix_response = openrouteservice.distance_matrix.distance_matrix(client,
                                                                                request_coordinates,
                                                                                profile='driving-hgv',
                                                                                sources=[0],
                                                                                destinations=list(range(1, 100)),
                                                                                metrics=['distance'],
                                                                                units='km')
    print('finished')
    for count, destination_location in enumerate(reduced_location_coordinates):
        distances[source_location][destination_location] = distance_matrix_response['distances'][0][count]
    sleep(2)  # Sleeping to not stretch openrouteservice api limit

