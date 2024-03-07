from flask import Flask, render_template
from datetime import datetime
import pytz
import csv
import wait_times

app = Flask(__name__, instance_relative_config=True)

rides = {
	"pirates_of_caribbean" : {
		"name": "Pirates of the Caribbean"
	}
}

def get_cur_time():
	eastern_timezone = pytz.timezone('US/Eastern')
	return datetime.now(eastern_timezone)
	return current_time_et.strftime('%H:%M:%S')

def get_all_predictions(day):
	predictions = {}
	for ride in rides.keys():
		predictions[ride] = get_predictions(ride, day)
	return predictions

def get_predictions(ride, day):
	predictions = {}
	day_int = wait_times.date_to_int(day)
	for model in ["dense50"]:
		with open(f"predictions/{ride}_{day_int}_{model}.csv", "r") as csvfile:
			ride_predicts = csv.reader(csvfile)
			next(ride_predicts)
			for row in ride_predicts:
				if row[1] not in predictions:
					predictions[row[1]] = {}
				predictions[row[1]][model] = int(float(row[2]))

	flat_predictions = []
	for time, models in predictions.items():
		p = {"time": time}
		for model, wait in models.items():
			p[model] = wait
		flat_predictions.append(p)
	return flat_predictions

@app.route("/")
def index():
	return render_template("index.html")

@app.route("/details/<string:ride>")
def details(ride):
	day = "3/10/2024" # TODO: get today. if it's not in the range of predictions, just use 310
	predictions = get_predictions(ride, day)
	return render_template("ride.html", ride=rides[ride]["name"], day=day, predictions=predictions)

if __name__ == "__main__":
	app.run()
