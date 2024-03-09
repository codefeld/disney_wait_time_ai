import numpy as np
import wait_times
from sklearn.preprocessing import StandardScaler

def prep_wait_times(ride_name):
	data_file = f"data/{ride_name}.csv"
	data = wait_times.parse_wait_times(data_file)
	prepped_data = {}
	for key, value in data.items():
		prepped_data[key] = np.array(value).reshape(-1, 1)
	return prepped_data

def fit_wait_times(ride_name):
	data = prep_wait_times(ride_name)

	# Normalize data
	fitted_data = {}
	for key, value in data.items():
		scaler = StandardScaler()
		scaler.fit(value)
		fitted_data[key] = scaler
	return fitted_data

def fit_transform_wait_times(ride_name):
	data = prep_wait_times(ride_name)

	# Normalize data
	fitted_data = {}
	for key, value in data.items():
		scaler = StandardScaler()
		fitted_data[key] = scaler.fit_transform(value)
	return fitted_data
