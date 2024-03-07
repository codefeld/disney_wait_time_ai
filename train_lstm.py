import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import wait_times

seq_length = 5  # Number of previous time steps to consider

rides = {
	"pirates_of_caribbean": 1,
	"dinosaur": 2,
}

def train_model(ride):
	data_file = f"data/{ride}.csv"
	dates, daysofweek, times, waits = wait_times.parse_wait_times(data_file)
	# ride_name = np.full(len(dates), rides[ride])
	ride_dates = np.array(dates)
	ride_daysofweek = np.array(daysofweek)
	ride_times = np.array(times)
	ride_durations = np.array(waits)

	# Normalize data
	#scaler_names = StandardScaler()
	scaler_dates = StandardScaler()
	scaler_daysofweek = StandardScaler()
	scaler_times = StandardScaler()
	scaler_durations = StandardScaler()

	#ride_names_scaled = scaler_names.fit_transform(np.array(rides.values()).reshape(-1, 1))
	ride_dates_scaled = scaler_dates.fit_transform(ride_dates.reshape(-1, 1))
	ride_daysofweek_scaled = scaler_daysofweek.fit_transform(ride_daysofweek.reshape(-1, 1))
	ride_times_scaled = scaler_times.fit_transform(ride_times.reshape(-1, 1))
	ride_durations_scaled = scaler_durations.fit_transform(ride_durations.reshape(-1, 1))

	# Combine dates and times into one input array
	# these are the colums we'll use as input when we use the model to make a prediction after it's trained
	X = np.column_stack((ride_dates_scaled, ride_daysofweek_scaled, ride_times_scaled))

	# Create sequences for LSTM
	X_seq, y_seq = [], []
	for i in range(len(ride_dates_scaled) - seq_length):
			X_seq.append(X[i:i + seq_length])
			y_seq.append(ride_durations_scaled[i + seq_length])

	X_seq = np.array(X_seq)
	y_seq = np.array(y_seq)

	# Split data into training and testing sets
	X_train, X_test, y_train, y_test = train_test_split(
			X_seq, # these are the parameters we'll use as input when we want to make a prediction after the model is trained
			y_seq, # these are actual result value we'll use to train the model on. they are equivilent to the value we want to predict
			test_size=0.2,
			random_state=42
	)

	model = tf.keras.Sequential([
			tf.keras.layers.Input(shape=(seq_length,3)),
			tf.keras.layers.LSTM(50, activation='relu'),
			tf.keras.layers.Dense(1, activation='linear')
	])

	# Compile the model
	model.compile(optimizer='adam', loss='mean_squared_error')

	# Train the model
	model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=1)

	model.summary()
	model.save(f"models/{ride}_lstm50.keras")

	# Evaluate the model on the test set
	loss = model.evaluate(X_test, y_test)
	print(f'Mean Squared Error on Test Set: {loss}')

if __name__ == "__main__":
	train_model("pirates_of_caribbean")
	# train_model("dinosaur")
