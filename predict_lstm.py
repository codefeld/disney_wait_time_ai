import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import wait_times
import csv
from datetime import datetime

def predict_wait_times(ride, day, start_hour, end_hour):
	data_file = f"data/{ride}.csv"
	dates, daysofweek, times, waits = wait_times.parse_wait_times(data_file)
	ride_dates = np.array(dates)
	ride_daysofweek = np.array(daysofweek)
	ride_times = np.array(times)
	ride_waits = np.array(waits)

	# Normalize data
	scaler_dates = StandardScaler()
	scaler_daysofweek = StandardScaler()
	scaler_times = StandardScaler()
	scaler_durations = StandardScaler()

	scaler_dates.fit(ride_dates.reshape(-1, 1))
	scaler_daysofweek.fit(ride_daysofweek.reshape(-1, 1))
	scaler_times.fit(ride_times.reshape(-1, 1))
	scaler_durations.fit(ride_waits.reshape(-1, 1))

	seq_length = 5  # Number of previous time steps to consider
	model = keras.models.load_model(f"models/{ride}_lstm50.keras")

	# Normalize future event data
	predict_dates = []
	predict_daysofweek = []
	predict_times = []

	date_object = datetime.strptime(day, "%m/%d/%Y")
	day_of_week = date_object.weekday() + 1
	day_int = wait_times.date_to_int(day)
	for hour in range(start_hour-1, end_hour): # subtract 1 from the hour so that we have elements for the LSTM sequences
		predict_dates.append(day_int)
		predict_dates.append(day_int)
		predict_dates.append(day_int)
		predict_dates.append(day_int)
		predict_daysofweek.append(day_of_week)
		predict_daysofweek.append(day_of_week)
		predict_daysofweek.append(day_of_week)
		predict_daysofweek.append(day_of_week)
		predict_times.append(wait_times.time_to_int(f"{hour}:00:00"))
		predict_times.append(wait_times.time_to_int(f"{hour}:15:00"))
		predict_times.append(wait_times.time_to_int(f"{hour}:30:00"))
		predict_times.append(wait_times.time_to_int(f"{hour}:45:00"))

	# Normalize future event data
	future_event_dates_scaled = scaler_dates.transform(np.array(predict_dates).reshape(-1, 1))
	future_event_daysofweek_scaled = scaler_daysofweek.transform(np.array(predict_daysofweek).reshape(-1, 1))
	future_event_times_scaled = scaler_times.transform(np.array(predict_times).reshape(-1, 1))

	future_event_data_scaled = np.column_stack((future_event_dates_scaled, future_event_daysofweek_scaled, future_event_times_scaled))

	with open(f"predictions/{ride}_{day_int}_lstm50.csv", "w") as csv_file:
		csv_writer = csv.writer(csv_file)
		csv_writer.writerow(["date", "time", "wait"])

	# Start at index 4, which will be start_hour (because we added 4 elements in front of it)
	for i in range(4, len(predict_times)):
		# Create sequences for LSTM using future event data
		X_future_seq = np.array([future_event_data_scaled[i-4:i]]) # what does this do?

		# Predict future event duration
		predicted_duration_scaled = model.predict(X_future_seq)

		# Inverse transform to get the actual predicted duration
		predict_waits = scaler_durations.inverse_transform(predicted_duration_scaled)
		print(predict_waits)

		with open(f"predictions/{ride}_{day_int}_lstm50.csv", "a") as csv_file:
			csv_writer = csv.writer(csv_file)
			csv_writer.writerow([day, wait_times.int_to_time(predict_times[i]), predict_waits[0][0]])

if __name__ == "__main__":
	for mk_ride in ["7_dwarfs_train", "pirates_of_caribbean"]:
		predict_wait_times(mk_ride, "3/10/2024", 8, 23)
		predict_wait_times(mk_ride, "3/11/2024", 8, 23)
		predict_wait_times(mk_ride, "3/12/2024", 8, 23)
		predict_wait_times(mk_ride, "3/13/2024", 8, 24)
		predict_wait_times(mk_ride, "3/14/2024", 8, 23)
		predict_wait_times(mk_ride, "3/15/2024", 8, 23)
	for ep_ride in ["soarin", "spaceship_earth"]:
		predict_wait_times(ep_ride, "3/10/2024", 9, 21)
		predict_wait_times(ep_ride, "3/11/2024", 9, 23)
		predict_wait_times(ep_ride, "3/12/2024", 9, 9)
		predict_wait_times(ep_ride, "3/13/2024", 9, 9)
		predict_wait_times(ep_ride, "3/14/2024", 9, 9)
		predict_wait_times(ep_ride, "3/15/2024", 9, 9)
	for hs_ride in ["alien_saucers", "toy_story_mania", "slinky_dog"]:
		predict_wait_times(hs_ride, "3/10/2024", 8, 9)
		predict_wait_times(hs_ride, "3/11/2024", 8, 9)
		predict_wait_times(hs_ride, "3/12/2024", 8, 9)
		predict_wait_times(hs_ride, "3/13/2024", 8, 9)
		predict_wait_times(hs_ride, "3/14/2024", 8, 9)
		predict_wait_times(hs_ride, "3/15/2024", 8, 9)
	for ak_ride in ["dinosaur", "expedition_everest", "flight_of_passage", "kilimanjaro_safaris", "navi_river"]:
		predict_wait_times(ak_ride, "3/10/2024", 8, 8)
		predict_wait_times(ak_ride, "3/11/2024", 8, 8)
		predict_wait_times(ak_ride, "3/12/2024", 8, 8)
		predict_wait_times(ak_ride, "3/13/2024", 7, 8)
		predict_wait_times(ak_ride, "3/14/2024", 7, 8)
		predict_wait_times(ak_ride, "3/15/2024", 8, 8)
	
