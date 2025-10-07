import numpy as np
import wait_times
from sklearn.preprocessing import StandardScaler

def dates(): 
	return {
		"10/08/2025":{
			"precip": 0.2,
			"hightemp": 89,
			"lowtemp": 74,
			"mk_open": 9,
			"mk_close": 23,
			"ep_open": 9,
			"ep_close": 21,
			"hs_open": 9,
			"hs_close": 21,
			"ak_open": 8,
			"ak_close": 18
		},
		"10/09/2025": {
			"precip": 0.61,
			"hightemp": 85,
			"lowtemp": 73,
			"mk_open": 8,
			"mk_close": 18,
			"ep_open": 9,
			"ep_close": 21,
			"hs_open": 9,
			"hs_close": 21,
			"ak_open": 8,
			"ak_close": 18
		},
		"10/10/2025": {
			"precip": 0.69,
			"hightemp": 82,
			"lowtemp": 70,
			"mk_open": 8,
			"mk_close": 18,
			"ep_open": 9,
			"ep_close": 21,
			"hs_open": 9,
			"hs_close": 21,
			"ak_open": 8,
			"ak_close": 19
		},
		"10/11/2025": {
			"precip": 0.24,
			"hightemp": 81,
			"lowtemp": 66,
			"mk_open": 9,
			"mk_close": 23,
			"ep_open": 9,
			"ep_close": 21,
			"hs_open": 9,
			"hs_close": 21,
			"ak_open": 8,
			"ak_close": 19
		}
	}

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
