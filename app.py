from flask import Flask, render_template
from datetime import datetime
import pytz
import csv
import wait_times

app = Flask(__name__, instance_relative_config=True)

models = ["dense50", "lstm50"]

rides = {
	"pirates_of_caribbean" : {
		"name": "Pirates of the Caribbean"
	}
}

def get_cur_time():
	eastern_timezone = pytz.timezone('US/Eastern')
	return datetime.now(eastern_timezone)
	#return current_time_et.strftime('%H:%M:%S')

def get_all_predictions(day, time):
	predictions_by_ride = []
	for ride in rides.keys():
		p = get_predictions(ride, day)
		p_at_time = p[time]
		new_p = {
			"name": rides[ride]['name'],
			"key": ride,
		}
		for model in models:
			new_p[model] = p_at_time[model]
		predictions_by_ride.append(new_p)
	return predictions_by_ride

def get_predictions(ride, day):
	predictions = {}
	day_int = wait_times.date_to_int(day)
	for model in models:
		with open(f"predictions/{ride}_{day_int}_{model}.csv", "r") as csvfile:
			ride_predicts = csv.reader(csvfile)
			next(ride_predicts)
			for row in ride_predicts:
				if row[1] not in predictions:
					predictions[row[1]] = {}
				predictions[row[1]][model] = int(float(row[2]))
	return predictions

def flatten_predictions(predictions):
	flat_predictions = []
	for time, models in predictions.items():
		p = {"time": time}
		for model, wait in models.items():
			p[model] = wait
		flat_predictions.append(p)
	return flat_predictions

@app.route("/")
def index():
	day = "3/10/2024" # TODO: get today. if it's not in the range of predictions, just use 310
	time = "12:00:00"
	predictions = get_all_predictions(day, time)
	return render_template("index.html", day=day, time=time, predictions=predictions)

@app.route("/details/<string:ride>")
def details(ride):
	day = "3/10/2024" # TODO: get today. if it's not in the range of predictions, just use 310
	predictions = get_predictions(ride, day)
	flat_predictions = flatten_predictions(predictions)
	return render_template("ride.html", ride=rides[ride]["name"], day=day, predictions=flat_predictions)

if __name__ == "__main__":
	app.run()
