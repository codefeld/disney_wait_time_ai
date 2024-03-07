import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import wait_times

def train_model(ride):
	data_file = f"data/{ride}.csv"
	dates, _, times, waits = wait_times.parse_wait_times(data_file)
	ride_dates = np.array(dates)
	ride_times = np.array(times)
	ride_waits = np.array(waits)

	# Normalize data
	scaler_dates = StandardScaler()
	scaler_times = StandardScaler()
	scaler_durations = StandardScaler()

	ride_dates_scaled = scaler_dates.fit_transform(ride_dates.reshape(-1, 1))
	ride_times_scaled = scaler_times.fit_transform(ride_times.reshape(-1, 1))
	ride_waits_scaled = scaler_durations.fit_transform(ride_waits.reshape(-1, 1))

	# Combine dates and times into one input array
	X = np.column_stack((ride_dates_scaled, ride_times_scaled))

	# Split data into training and testing sets
	X_train, X_test, y_train, y_test = train_test_split(
		X, ride_waits_scaled, test_size=0.2, random_state=42
	)

	# Build the neural network model
	model = tf.keras.Sequential([
		tf.keras.layers.Input(shape=(2,)),
		tf.keras.layers.Dense(50, activation='relu'),
		tf.keras.layers.Dense(1, activation='linear')
	])

	# Compile the model
	model.compile(optimizer='adam', loss='mean_squared_error')

	# Train the model
	model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=1)

	model.summary()
	model.save(f"models/{ride}_dense50.keras")

	# Evaluate the model on the test set
	loss = model.evaluate(X_test, y_test)
	print(f'Mean Squared Error on Test Set: {loss}')

if __name__ == "__main__":
	train_model("7_dwarfs_train")
	train_model("alien_saucers")
	train_model("dinosaur")
	train_model("expedition_everest")
	train_model("flight_of_passage")
	train_model("kilimanjaro_safaris")
	train_model("navi_river")
	train_model("rock_n_rollercoaster")
	train_model("slinky_dog")
	train_model("soarin")
	train_model("spaceship_earth")
	train_model("toy_story_mania")
