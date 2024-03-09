import os
import csv
import math
from datetime import datetime, timedelta
import numpy as np
from sklearn.preprocessing import StandardScaler

def date_to_int(datestr):
	date_parts = datestr.split("/")
	# return int(year) * 10000 + int(month) * 100 + int(day)
	return int(date_parts[0]) * 100 + int(date_parts[1]) # let's ignore the year (for now...)

def int_to_date(date):
	return f"{math.floor(date / 100)}/{date % 100}"

def time_to_int(timestr):
	time_object = datetime.strptime(timestr, "%H:%M:%S")
	return (time_object - datetime(1971, 10, 1)).seconds

def int_to_time(time_int):
	new_time_object = datetime(1971, 10, 1) + timedelta(seconds=time_int)
	return new_time_object.strftime('%H:%M:%S')

def prep_wait_times(ride_name, method_name):
	data_file = f"data/{ride_name}.csv"
	data = parse_wait_times(data_file)
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

def parse_wait_times(filename):
	dates = []
	daysofweek = []
	hightemps = []
	lowtemps = []
	precip = []
	times = []
	waits = []
	metadata = {}
	with open("data/metadata.csv", "r") as csvfile:
		csv_reader = csv.DictReader(csvfile, delimiter=",")
		for row in csv_reader:
			metadata[row['DATE']] = {
				'dayofweek': row["DAYOFWEEK"],
				'precip': row["WEATHER_WDWPRECIP"],
				'high_temp': row["WEATHER_WDWHIGH"],
				'low_temp': row["WEATHER_WDWLOW"]
			}
	with open(filename, "r") as csvfile:
		csv_reader = csv.DictReader(csvfile, delimiter=",")
		for row in csv_reader:
			if row["SPOSTMIN"] and int(row["SPOSTMIN"]) >= 0 and row["date"] in metadata:
				dates.append(date_to_int(row["date"]))
				daysofweek.append(int(metadata[row["date"]]['dayofweek']))
				precip.append(float(metadata[row["date"]]["precip"]))
				hightemps.append(float(metadata[row["date"]]["high_temp"]))
				lowtemps.append(float(metadata[row["date"]]["low_temp"]))
				times.append(time_to_int(row["datetime"].split()[1]))
				waits.append(int(row["SPOSTMIN"]))
	return {
		"dates": dates,
		"daysofweek": daysofweek,
		"precip": precip,
		"hightemps": hightemps,
		"lowtemps": lowtemps,
		"times": times,
		"waits": waits
	}



