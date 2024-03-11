import numpy as np
import wait_times
from sklearn.preprocessing import StandardScaler

def dates(): 
	return {
		"3/10/2024":{
			"precip": 0.4,
			"hightemp": 76,
			"lowtemp": 62,
			"mk_open": 8,
			"mk_close": 23,
			"ep_open": 9,
			"ep_close": 21,
			"hs_open": 8,
			"hs_close": 21,
			"ak_open": 8,
			"ak_close": 20
		},
		"3/11/2024": {
			"precip": 0,
			"hightemp": 73,
			"lowtemp": 52,
			"mk_open": 8,
			"mk_close": 23,
			"ep_open": 9,
			"ep_close": 23,
			"hs_open": 8,
			"hs_close": 21,
			"ak_open": 8,
			"ak_close": 20
		},
		"3/12/2024": {
			"precip": 0,
			"hightemp": 78,
			"lowtemp": 53,
			"mk_open": 8,
			"mk_close": 23,
			"ep_open": 9,
			"ep_close": 21,
			"hs_open": 8,
			"hs_close": 21,
			"ak_open": 8,
			"ak_close": 20
		},
		"3/13/2024": {
			"precip": 0.15,
			"hightemp": 76,
			"lowtemp": 60,
			"mk_open": 8,
			"mk_close": 24,
			"ep_open": 9,
			"ep_close": 21,
			"hs_open": 8,
			"hs_close": 21,
			"ak_open": 7,
			"ak_close": 20
		},
		"3/14/2024": {
			"precip": 0,
			"hightemp": 82,
			"lowtemp": 61,
			"mk_open": 8,
			"mk_close": 23,
			"ep_open": 9,
			"ep_close": 21,
			"hs_open": 8,
			"hs_close": 21,
			"ak_open": 7,
			"ak_close": 20
		},
		"3/15/2024": {
			"precip": 0,
			"hightemp": 85,
			"lowtemp": 64,
			"mk_open": 8,
			"mk_close": 23,
			"ep_open": 9,
			"ep_close": 21,
			"hs_open": 8,
			"hs_close": 21,
			"ak_open": 8,
			"ak_close": 20
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
