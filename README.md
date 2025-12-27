# Disney Wait Time AI

A machine learning application that predicts wait times for popular Disney World attractions using historical data and weather patterns.

## Overview

This project uses TensorFlow/Keras neural networks to forecast wait times across multiple Disney World parks (Magic Kingdom, Epcot, Hollywood Studios, and Animal Kingdom). The web interface displays predictions for various rides throughout the day, helping visitors plan their park experience more effectively.

## Features

- **Multiple ML Models**: Supports various neural network architectures (Dense, LSTM, Weather-enhanced models)
- **12 Popular Attractions**: Predicts wait times for major rides across all four Disney World parks
- **Time-Series Predictions**: Generates forecasts in 15-minute intervals throughout park operating hours
- **Weather Integration**: Incorporates precipitation and temperature data for improved accuracy
- **Interactive Web Interface**: Flask-based UI for viewing predictions by time and ride
- **Heroku-Ready**: Configured for easy cloud deployment

## Supported Attractions

- **Magic Kingdom**: Pirates of the Caribbean, Seven Dwarfs Mine Train
- **Epcot**: Soarin' Around the World, Spaceship Earth
- **Hollywood Studios**: Alien Swirling Saucers, Slinky Dog Dash, Toy Story Mania!
- **Animal Kingdom**: DINOSAUR, Expedition Everest, Flight of Passage, Kilimanjaro Safaris, Na'vi River Journey

## How It Works

The system uses historical wait time data combined with metadata (day of week, weather conditions, park hours) to train neural network models. These models learn patterns in crowd behavior and predict future wait times based on similar conditions.

### Model Types

- `dense50`: Basic dense neural network with 50 neurons
- `densew2`: Weather-enhanced dense network
- `lstm50`: LSTM network for sequential pattern recognition
- `lstmw50`: Weather-enhanced LSTM network

## Installation

### Prerequisites

- Python 3.13+
- uv (Python package manager)

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd disney_wait_time_ai

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

## Usage

### Training Models

Train models for specific rides:

```bash
# Basic dense model
python train.py

# LSTM model
python train_lstm.py

# Weather-enhanced models
python train_lstmw.py
python train_densew.py
```

### Generating Predictions

Create prediction files for specific dates:

```bash
# Basic predictions
python predict.py

# LSTM predictions
python predict_lstm.py
python predict_lstmw.py

# Weather-enhanced predictions
python predict_densew.py
```

Predictions are saved to the `predictions/` directory as CSV files.

### Running the Web Application

Start the Flask development server:

```bash
python app.py
```

Or use gunicorn for production:

```bash
gunicorn app:app
```

The application will be available at `http://localhost:5000`

### Deployment to Heroku

```bash
heroku create
git push heroku main
```

## Project Structure

```
disney_wait_time_ai/
├── app.py                 # Flask web application
├── wait_times.py          # Data parsing and utility functions
├── ml_helper.py           # ML preprocessing helpers
├── train*.py              # Model training scripts
├── predict*.py            # Prediction generation scripts
├── data/                  # Historical wait time and weather data
│   ├── *_ride.csv         # Ride-specific wait time data
│   └── metadata.csv       # Weather and operational data
├── models/                # Trained Keras models
├── predictions/           # Generated prediction CSV files
├── templates/             # HTML templates
├── static/                # CSS/JS/images
├── Procfile              # Heroku deployment configuration
├── requirements.txt       # Python dependencies
└── pyproject.toml        # Project metadata
```

## Technologies Used

- **Python 3.13**: Core programming language
- **TensorFlow/Keras**: Neural network framework
- **scikit-learn**: Data preprocessing and evaluation
- **Flask**: Web framework
- **NumPy**: Numerical computations
- **Pandas**: Data manipulation
- **Gunicorn**: Production WSGI server
- **pytz**: Timezone handling

## Data Sources

The project uses historical wait time data collected from Disney World attractions, combined with:
- Day of week information
- Weather data (precipitation, high/low temperatures)
- Park operating hours
- Historical wait times sampled at regular intervals

## Future Enhancements

- Real-time data integration
- Additional parks and attractions
- Mobile-responsive design improvements
- API endpoints for third-party integrations
- Model performance comparisons and selection

## License

This project is for educational and personal use.
