import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from datetime import datetime
import wait_times
import csv
import ml_helper

model_name = "densew2"

def predict_wait_times(ride, scalers, date, precip, hightemp, lowtemp, start_hour, end_hour):
	model = keras.models.load_model(f"models/{ride}_{model_name}.keras")

	# Normalize future event data
	predict_dates = []
	predict_daysofweek = []
	predict_precip = []
	predict_hightemps = []
	predict_lowtemps = []
	predict_times = []

	date_object = datetime.strptime(date, "%m/%d/%Y")
	day_of_week = date_object.weekday() + 1
	day_int = wait_times.date_to_int(date)
	for hour in range(start_hour, end_hour):
		for i in range(4):
			predict_dates.append(day_int)
			predict_daysofweek.append(day_of_week)
			predict_precip.append(precip)
			predict_hightemps.append(hightemp)
			predict_lowtemps.append(lowtemp)
		predict_times.append(wait_times.time_to_int(f"{hour}:00:00"))
		predict_times.append(wait_times.time_to_int(f"{hour}:15:00"))
		predict_times.append(wait_times.time_to_int(f"{hour}:30:00"))
		predict_times.append(wait_times.time_to_int(f"{hour}:45:00"))

	# # Combine future event date and time
	# future_event_data_scaled = np.column_stack((future_event_date_scaled, future_event_time_scaled))
	future_event_dates_scaled = scalers["dates"].transform(np.array(predict_dates).reshape(-1, 1))
	future_event_daysofweek_scaled = scalers["daysofweek"].transform(np.array(predict_daysofweek).reshape(-1, 1))
	future_event_precip_scaled = scalers["precip"].transform(np.array(predict_precip).reshape(-1, 1))
	future_event_hightemp_scaled = scalers["hightemps"].transform(np.array(predict_hightemps).reshape(-1, 1))
	future_event_lowtemp_scaled = scalers["lowtemps"].transform(np.array(predict_lowtemps).reshape(-1, 1))
	future_event_times_scaled = scalers["times"].transform(np.array(predict_times).reshape(-1, 1))

	future_event_data_scaled = np.column_stack((future_event_dates_scaled, future_event_daysofweek_scaled, future_event_hightemp_scaled, future_event_lowtemp_scaled, future_event_times_scaled))

	# Predict future event duration
	predicted_duration_scaled = model.predict(future_event_data_scaled)

	# Inverse transform to get the actual predicted duration
	predict_wait = scalers["waits"].inverse_transform(predicted_duration_scaled)
	# print(predicted_duration)
	# print(f"Predicted Wait Time for {ride}:")
	# print(f"Date: {future_event_date.flatten()[0]}, Time: {future_event_time.flatten()[0]}, Predicted Wait Time: {predicted_duration.flatten()[0]:.2f} minutes")
	with open(f"predictions/{ride}_{day_int}_{model_name}.csv", "w") as csv_file:
		csv_writer = csv.writer(csv_file)
		csv_writer.writerow(["date", "time", "wait"])
		for i in range(len(predict_dates)):
			csv_writer.writerow([date, wait_times.int_to_time(predict_times[i]), predict_wait[i][0]])

if __name__ == "__main__":
	for mk_ride in ["7_dwarfs_train", "pirates_of_caribbean"]:
		scalers = ml_helper.fit_wait_times(mk_ride)
		predict_wait_times(mk_ride, scalers, "3/10/2024", 0.4, 76, 62, 8, 23)
		predict_wait_times(mk_ride, scalers, "3/11/2024", 0, 73, 52, 8, 23)
		predict_wait_times(mk_ride, scalers, "3/12/2024", 0, 78, 53, 8, 23)
		predict_wait_times(mk_ride, scalers, "3/13/2024", 0.15, 76, 60, 8, 24)
		predict_wait_times(mk_ride, scalers, "3/14/2024", 0, 82, 61, 8, 23)
		predict_wait_times(mk_ride, scalers, "3/15/2024", 0, 85, 64, 8, 23)
	for ep_ride in ["soarin", "spaceship_earth"]:
		scalers = ml_helper.fit_wait_times(ep_ride)
		predict_wait_times(ep_ride, scalers, "3/10/2024", 0.4, 76, 62, 9, 21)
		predict_wait_times(ep_ride, scalers, "3/11/2024", 0, 73, 52, 9, 23)
		predict_wait_times(ep_ride, scalers, "3/12/2024", 0, 78, 53, 9, 21)
		predict_wait_times(ep_ride, scalers, "3/13/2024", 0.15, 76, 60, 9, 21)
		predict_wait_times(ep_ride, scalers, "3/14/2024", 0, 82, 61, 9, 21)
		predict_wait_times(ep_ride, scalers, "3/15/2024", 0, 85, 64, 9, 21)
	for hs_ride in ["alien_saucers", "toy_story_mania", "slinky_dog"]:
		scalers = ml_helper.fit_wait_times(hs_ride)
		predict_wait_times(hs_ride, scalers, "3/10/2024", 0.4, 76, 62, 8, 21)
		predict_wait_times(hs_ride, scalers, "3/11/2024", 0, 73, 52, 8, 21)
		predict_wait_times(hs_ride, scalers, "3/12/2024", 0, 78, 53, 8, 21)
		predict_wait_times(hs_ride, scalers, "3/13/2024", 0.15, 76, 60, 8, 21)
		predict_wait_times(hs_ride, scalers, "3/14/2024", 0, 82, 61, 8, 21)
		predict_wait_times(hs_ride, scalers, "3/15/2024", 0, 85, 64, 8, 21)
	for ak_ride in ["dinosaur", "expedition_everest", "flight_of_passage", "kilimanjaro_safaris", "navi_river"]:
		scalers = ml_helper.fit_wait_times(ak_ride)
		predict_wait_times(ak_ride, scalers, "3/10/2024", 0.4, 76, 62, 8, 20)
		predict_wait_times(ak_ride, scalers, "3/11/2024", 0, 73, 52, 8, 20)
		predict_wait_times(ak_ride, scalers, "3/12/2024", 0, 78, 53, 8, 20)
		predict_wait_times(ak_ride, scalers, "3/13/2024", 0.15, 76, 60, 7, 20)
		predict_wait_times(ak_ride, scalers, "3/14/2024", 0, 82, 61, 7, 20)
		predict_wait_times(ak_ride, scalers, "3/15/2024", 0, 85, 64, 8, 20)
