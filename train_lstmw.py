import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import ml_helper

seq_length = 5  # Number of previous time steps to consider

def train_model(ride):
	print(f"[Training {ride} - lstmw]")
	data = ml_helper.fit_transform_wait_times(ride)

	# Combine dates and times into one input array
	# these are the columns we'll use as input when we use the model to make a prediction after it's trained
	X = np.column_stack((data["dates"], data["daysofweek"], data["hightemps"], data["lowtemps"], data["times"]))

	# Create sequences for LSTM
	X_seq, y_seq = [], []
	for i in range(len(data["dates"]) - seq_length):
			X_seq.append(X[i:i + seq_length])
			y_seq.append(data["waits"][i + seq_length])

	X_seq = np.array(X_seq)
	y_seq = np.array(y_seq)

	# Split data into training and testing sets
	X_train, X_test, y_train, y_test = train_test_split(
			X_seq, # these are the parameters we'll use as input when we want to make a prediction after the model is trained
			y_seq, # these are actual result values we'll use to train the model on. they are equivalent to the value we want to predict
			test_size=0.2,
			random_state=42
	)

	model = tf.keras.Sequential([
			tf.keras.layers.Input(shape=(seq_length,5)),
			tf.keras.layers.LSTM(50, activation='relu'),
			tf.keras.layers.Dense(1, activation='linear')
	])

	# Compile the model
	model.compile(optimizer='adam', loss='mean_squared_error')

	# Train the model
	model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=1)

	model.summary()
	model.save(f"models/{ride}_lstmw50.keras")

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
	train_model("pirates_of_caribbean")
	train_model("rock_n_rollercoaster")
	train_model("slinky_dog")
	train_model("soarin")
	train_model("spaceship_earth")
	train_model("toy_story_mania")
