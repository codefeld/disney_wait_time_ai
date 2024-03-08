import os
import csv
import math
from datetime import datetime, timedelta

def date_to_int(datestr):
	date_parts = datestr.split("/")
	# return int(year) * 10000 + int(month) * 100 + int(day)
	return int(date_parts[0]) * 100 + int(date_parts[1]) # let's ignore the year (for now...)

def int_to_date(date):
	return f"{math.floor(date / 100)}/{date % 100}"

def time_to_int(timestr):
	time_object = datetime.strptime(timestr, "%H:%M:%S")
	return (time_object - datetime(1971, 10, 1)).seconds

def int_to_time(time_int):
	new_time_object = datetime(1971, 10, 1) + timedelta(seconds=time_int)
	return new_time_object.strftime('%H:%M:%S')


def parse_wait_times(filename):
	dates = []
	daysofweek = []
	times = []
	waits = []
	metadata = {}
	with open("data/metadata.csv", "r") as csvfile:
		csv_reader = csv.DictReader(csvfile, delimiter=",")
		for row in csv_reader:
			metadata[row['DATE']] = {
				'dayofweek': 0,
			}
	with open(filename, "r") as csvfile:
		csv_reader = csv.DictReader(csvfile, delimiter=",")
		for row in csv_reader:
			if row["SPOSTMIN"] and int(row["SPOSTMIN"]) >= 0 and row["date"] in metadata:
				dates.append(date_to_int(row["date"]))
				daysofweek.append(int(metadata[row["date"]]['dayofweek']))
				times.append(time_to_int(row["datetime"].split()[1]))
				waits.append(int(row["SPOSTMIN"]))
	return dates, daysofweek, times, waits



