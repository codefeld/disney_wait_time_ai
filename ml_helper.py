import numpy as np
import wait_times
from sklearn.preprocessing import StandardScaler

def prep_wait_times(ride_name, method_name):
	data_file = f"data/{ride_name}.csv"
	data = wait_times.parse_wait_times(data_file)
	ride_dates = np.array(data["dates"])
	ride_daysofweek = np.array(data["daysofweek"])
	ride_precip = np.array(data["precip"])
	ride_hightemp = np.array(data["hightemps"])
	ride_lowtemp = np.array(data["lowtemps"])
	ride_times = np.array(data["times"])
	ride_durations = np.array(data["waits"])

	# Normalize data
	#scaler_names = StandardScaler()
	scaler_dates = StandardScaler()
	scaler_daysofweek = StandardScaler()
	scaler_precip = StandardScaler()
	scaler_hightemp = StandardScaler()
	scaler_lowtemp = StandardScaler()
	scaler_times = StandardScaler()
	scaler_durations = StandardScaler()

	#ride_names_scaled = scaler_names.fit_transform(np.array(rides.values()).reshape(-1, 1))
	ride_dates_scaled = getattr(scaler_dates, method_name)(ride_dates.reshape(-1, 1))
	ride_daysofweek_scaled = getattr(scaler_daysofweek, method_name)(ride_daysofweek.reshape(-1, 1))
	ride_precip_scaled = getattr(scaler_precip, method_name)(ride_precip.reshape(-1, 1))
	ride_hightemp_scaled = getattr(scaler_hightemp, method_name)(ride_hightemp.reshape(-1, 1))
	ride_lowtemp_scaled = getattr(scaler_lowtemp, method_name)(ride_lowtemp.reshape(-1, 1))
	ride_times_scaled = getattr(scaler_times, method_name)(ride_times.reshape(-1, 1))
	ride_durations_scaled = getattr(scaler_durations, method_name)(ride_durations.reshape(-1, 1))
	return {
		"dates": ride_dates_scaled,
		"daysofweek": ride_daysofweek_scaled,
		"precip": ride_precip_scaled,
		"hightemps": ride_hightemp_scaled,
		"lowtemps": ride_lowtemp_scaled,
		"times": ride_times_scaled,
		"waits": ride_durations_scaled
	}

def transform_wait_times(ride_name):
	return prep_wait_times(ride_name, "transform")

def fit_transform_wait_times(ride_name):
	return prep_wait_times(ride_name, "fit_transform")