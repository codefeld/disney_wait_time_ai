import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import wait_times
import csv
from datetime import datetime
import ml_helper

def predict_wait_times(ride, scalers, day, precip, hightemp, lowtemp, start_hour, end_hour):
	model = keras.models.load_model(f"models/{ride}_lstmw50.keras")

	# Normalize future event data
	predict_dates = []
	predict_daysofweek = []
	predict_precip = []
	predict_hightemp = []
	predict_lowtemp = []
	predict_times = []

	date_object = datetime.strptime(day, "%m/%d/%Y")
	day_of_week = date_object.weekday() + 1
	day_int = wait_times.date_to_int(day)
	for hour in range(start_hour-1, end_hour): # subtract 1 from the hour so that we have elements for the LSTM sequences
		for i in range(4):
			predict_dates.append(day_int)
			predict_daysofweek.append(day_of_week)
			predict_precip.append(precip)
			predict_hightemp.append(hightemp)
			predict_lowtemp.append(lowtemp)
		predict_times.append(wait_times.time_to_int(f"{hour}:00:00"))
		predict_times.append(wait_times.time_to_int(f"{hour}:15:00"))
		predict_times.append(wait_times.time_to_int(f"{hour}:30:00"))
		predict_times.append(wait_times.time_to_int(f"{hour}:45:00"))

	# Normalize future event data
	future_event_dates_scaled = scalers["dates"].transform(np.array(predict_dates).reshape(-1, 1))
	future_event_daysofweek_scaled = scalers["daysofweek"].transform(np.array(predict_daysofweek).reshape(-1, 1))
	future_event_precip_scaled = scalers["precip"].transform(np.array(predict_precip).reshape(-1, 1))
	future_event_hightemp_scaled = scalers["hightemps"].transform(np.array(predict_hightemp).reshape(-1, 1))
	future_event_lowtemp_scaled = scalers["lowtemps"].transform(np.array(predict_lowtemp).reshape(-1, 1))
	future_event_times_scaled = scalers["times"].transform(np.array(predict_times).reshape(-1, 1))

	future_event_data_scaled = np.column_stack((future_event_dates_scaled, future_event_daysofweek_scaled, future_event_hightemp_scaled, future_event_lowtemp_scaled, future_event_times_scaled))

	with open(f"predictions/{ride}_{day_int}_lstmw50.csv", "w") as csv_file:
		csv_writer = csv.writer(csv_file)
		csv_writer.writerow(["date", "time", "wait"])

	# Start at index 4, which will be start_hour (because we added 4 elements in front of it)
	for i in range(4, len(predict_times)):
		# Create sequences for LSTM using future event data
		X_future_seq = np.array([future_event_data_scaled[i-4:i]]) # what does this do?

		# Predict future event duration
		predicted_duration_scaled = model.predict(X_future_seq)

		# Inverse transform to get the actual predicted duration
		predict_waits = scalers["waits"].inverse_transform(predicted_duration_scaled)
		print(predict_waits)

		with open(f"predictions/{ride}_{day_int}_lstmw50.csv", "a") as csv_file:
			csv_writer = csv.writer(csv_file)
			csv_writer.writerow([day, wait_times.int_to_time(predict_times[i]), predict_waits[0][0]])

if __name__ == "__main__":
	for mk_ride in ["7_dwarfs_train", "pirates_of_caribbean"]:
		scalers = ml_helper.fit_wait_times(mk_ride)
		for date, values in ml_helper.dates().items():
			predict_wait_times(mk_ride, scalers, date, values['precip'], values['hightemp'], values['lowtemp'], values['mk_open'], values['mk_close'])
	for ep_ride in ["soarin", "spaceship_earth"]:
		scalers = ml_helper.fit_wait_times(ep_ride)
		for date, values in ml_helper.dates().items():
			predict_wait_times(mk_ride, scalers, date, values['precip'], values['hightemp'], values['lowtemp'], values['ep_open'], values['ep_close'])
	for hs_ride in ["alien_saucers", "toy_story_mania", "slinky_dog"]:
		scalers = ml_helper.fit_wait_times(hs_ride)
		for date, values in ml_helper.dates().items():
			predict_wait_times(mk_ride, scalers, date, values['precip'], values['hightemp'], values['lowtemp'], values['hs_open'], values['hs_close'])
	for ak_ride in ["dinosaur", "expedition_everest", "flight_of_passage", "kilimanjaro_safaris", "navi_river"]:
		scalers = ml_helper.fit_wait_times(ak_ride)
		for date, values in ml_helper.dates().items():
			predict_wait_times(mk_ride, scalers, date, values['precip'], values['hightemp'], values['lowtemp'], values['ak_open'], values['ak_close'])
	

