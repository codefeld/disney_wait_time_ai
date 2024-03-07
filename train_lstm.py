import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import wait_times

def train_model(ride):
	data_file = f"data/{ride}.csv"
	dates, times, waits = wait_times.parse_wait_times(data_file)
	event_dates = np.array(dates)
	event_times = np.array(times)
	event_durations = np.array(waits)

	# Normalize data
	scaler_dates = StandardScaler()
	scaler_times = StandardScaler()
	scaler_durations = StandardScaler()

	event_dates_scaled = scaler_dates.fit_transform(event_dates.reshape(-1, 1))
	event_times_scaled = scaler_times.fit_transform(event_times.reshape(-1, 1))
	event_durations_scaled = scaler_durations.fit_transform(event_durations.reshape(-1, 1))

	# Combine dates and times into one input array
	# these are the colums we'll use as input when we use the model to make a prediction after it's trained
	X = np.column_stack((event_dates_scaled, event_times_scaled))

	# Create sequences for LSTM
	seq_length = 5  # Number of previous time steps to consider
	X_seq, y_seq = [], []
	for i in range(len(event_dates_scaled) - seq_length):
			X_seq.append(X[i:i + seq_length])
			y_seq.append(event_durations_scaled[i + seq_length])

	X_seq = np.array(X_seq)
	y_seq = np.array(y_seq)

	# Split data into training and testing sets
	X_train, X_test, y_train, y_test = train_test_split(
			X_seq, # these are the parameters we'll use as input when we want to make a prediction after the model is trained
			y_seq, # these are actual result value we'll use to train the model on. they are equivilent to the value we want to predict
			test_size=0.2,
			random_state=42
	)

	# Build the LSTM model
	model = tf.keras.Sequential([
			tf.keras.layers.Input(shape=(seq_length,2)),
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

