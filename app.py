from flask import Flask, render_template, request
from datetime import datetime, timedelta
import pytz
import csv
import wait_times
import os

app = Flask(__name__, instance_relative_config=True)

models = ["dense50", "lstm50", "lstmw50"]

rides = {
	"pirates_of_caribbean" : {
		"name": "Pirates of the Caribbean"
	},
	"7_dwarfs_train" : {
		"name": "Seven Dwarfs"
	}
}

def get_datetime():
	curdate = get_cur_time()
	day = "3/10/2024"
	time = "12:00:00"
	if curdate.month == 3 and curdate.year == 2024 and curdate.day in [10,11,12,13,14,15]:
		day = curdate.strftime('%m/%d/%Y')
		time = curdate.strftime('%H:00:00')
	return day, time

def get_cur_time():
	eastern_timezone = pytz.timezone('US/Eastern')
	return datetime.now(eastern_timezone)

def get_all_predictions(day, time):
	predictions_by_ride = []
	for ride in rides.keys():
		p = get_predictions(ride, day)
		if time in p:
			p_at_time = p[time]
			new_p = {
				"name": rides[ride]['name'],
				"key": ride,
			}
			for model in models:
				if model in p_at_time:
					new_p[model] = p_at_time[model]
			predictions_by_ride.append(new_p)
	return predictions_by_ride

def get_predictions(ride, day):
	predictions = {}
	day_int = wait_times.date_to_int(day)
	for model in models:
		pfile = f"predictions/{ride}_{day_int}_{model}.csv"
		if os.path.exists(pfile):
			with open(pfile, "r") as csvfile:
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

def get_prev_cur_next_day(day):
	d = datetime.strptime(day, "%m/%d/%Y")
	return (d - timedelta(days=1)).strftime("%d"), d.day, (d + timedelta(days=1)).strftime("%d")

def get_prev_and_next_time(time):
	t = datetime.strptime(time, "%H:%M:%S")
	return (t - timedelta(minutes=15)).strftime("%H:%M:%S"), (t + timedelta(minutes=15)).strftime("%H:%M:%S")

@app.route("/")
def index():
	day, time = get_datetime()
	day_param = request.args.get('day')
	if day_param:
		day = f"3/{day_param}/2024"
	time_param = request.args.get('time')
	if time_param:
		time = time_param
	prev_day, cur_day, next_day = get_prev_cur_next_day(day)
	prev_time, next_time = get_prev_and_next_time(time)
	predictions = get_all_predictions(day, time)
	return render_template(
		"index.html",
		day=day, prev_day=prev_day, next_day=next_day, cur_day=cur_day,
		time=time, prev_time=prev_time, next_time=next_time,
		predictions=predictions,
	)

@app.route("/details/<string:ride>")
def details(ride):
	day, _ = get_datetime()
	day_param = request.args.get('day')
	if day_param:
		day = f"3/{day_param}/2024"
	prev_day, cur_day, next_day = get_prev_cur_next_day(day)
	predictions = get_predictions(ride, day)
	flat_predictions = flatten_predictions(predictions)
	return render_template(
		"ride.html", ride=rides[ride]["name"], key=ride,
		day=day, prev_day=prev_day, next_day=next_day, cur_day=cur_day,
		predictions=flat_predictions)

if __name__ == "__main__":
	app.run()
