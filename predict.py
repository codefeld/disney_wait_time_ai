import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import wait_times
import csv

def predict_wait_times(ride, day, start_hour, end_hour):
	data_file = f"data/{ride}.csv"
	dates, _, times, waits = wait_times.parse_wait_times(data_file)
	ride_dates = np.array(dates)
	ride_times = np.array(times)
	ride_waits = np.array(waits)

	# Normalize data
	scaler_dates = StandardScaler()
	scaler_times = StandardScaler()
	scaler_durations = StandardScaler()

	scaler_dates.fit(ride_dates.reshape(-1, 1))
	scaler_times.fit(ride_times.reshape(-1, 1))
	scaler_durations.fit(ride_waits.reshape(-1, 1))

	model = keras.models.load_model(f"models/{ride}_dense50.keras")

	# Normalize future event data
	predict_dates = []
	predict_times = []
	day_int = wait_times.date_to_int(day)
	for hour in range(start_hour, end_hour):
		predict_dates.append(day_int)
		predict_dates.append(day_int)
		predict_dates.append(day_int)
		predict_dates.append(day_int)
		predict_times.append(wait_times.time_to_int(f"{hour}:00:00"))
		predict_times.append(wait_times.time_to_int(f"{hour}:15:00"))
		predict_times.append(wait_times.time_to_int(f"{hour}:30:00"))
		predict_times.append(wait_times.time_to_int(f"{hour}:45:00"))
	
	future_event_date = np.array(predict_dates).reshape(-1, 1)
	future_event_time = np.array(predict_times).reshape(-1, 1)

	future_event_date_scaled = scaler_dates.transform(future_event_date)
	future_event_time_scaled = scaler_times.transform(future_event_time)

	# Combine future event date and time
	future_event_data_scaled = np.column_stack((future_event_date_scaled, future_event_time_scaled))

	# Predict future event duration
	predicted_duration_scaled = model.predict(future_event_data_scaled)

	# Inverse transform to get the actual predicted duration
	predict_wait = scaler_durations.inverse_transform(predicted_duration_scaled)
	# print(predicted_duration)
	# print(f"Predicted Wait Time for {ride}:")
	# print(f"Date: {future_event_date.flatten()[0]}, Time: {future_event_time.flatten()[0]}, Predicted Wait Time: {predicted_duration.flatten()[0]:.2f} minutes")
	with open(f"predictions/{ride}_{day_int}_dense50.csv", "w") as csv_file:
		csv_writer = csv.writer(csv_file)
		csv_writer.writerow(["date", "time", "wait"])
		for i in range(len(predict_dates)):
			csv_writer.writerow([day, wait_times.int_to_time(predict_times[i]), predict_wait[i][0]])

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
		predict_wait_times(ep_ride, "3/12/2024", 9, 21)
		predict_wait_times(ep_ride, "3/13/2024", 9, 21)
		predict_wait_times(ep_ride, "3/14/2024", 9, 21)
		predict_wait_times(ep_ride, "3/15/2024", 9, 21)
	for hs_ride in ["alien_saucers", "toy_story_mania", "slinky_dog"]:
		predict_wait_times(hs_ride, "3/10/2024", 8, 21)
		predict_wait_times(hs_ride, "3/11/2024", 8, 21)
		predict_wait_times(hs_ride, "3/12/2024", 8, 21)
		predict_wait_times(hs_ride, "3/13/2024", 8, 21)
		predict_wait_times(hs_ride, "3/14/2024", 8, 21)
		predict_wait_times(hs_ride, "3/15/2024", 8, 21)
	for ak_ride in ["dinosaur", "expedition_everest", "flight_of_passage", "kilimanjaro_safaris", "navi_river"]:
		predict_wait_times(ak_ride, "3/10/2024", 8, 20)
		predict_wait_times(ak_ride, "3/11/2024", 8, 20)
		predict_wait_times(ak_ride, "3/12/2024", 8, 20)
		predict_wait_times(ak_ride, "3/13/2024", 7, 20)
		predict_wait_times(ak_ride, "3/14/2024", 7, 20)
		predict_wait_times(ak_ride, "3/15/2024", 8, 20)
	