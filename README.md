# FareSense India

> Ride fare estimates across 10 Indian cities — powered by machine learning.

Built with Python and Streamlit, FareSense predicts ride fares based on distance, time of day, traffic conditions, surge pricing, and vehicle type — calibrated from real Ola, Uber, and Rapido pricing data.

---

## Live Demo

[faresenseindia-divyajain.streamlit.app](https://fairsenseindia-divyajain.streamlit.app)

---

## Features

- Fare estimates for 10 cities: Delhi-NCR, Mumbai, Bangalore, Chennai, Hyderabad, Kolkata, Pune, Ahmedabad, Guwahati, Bhopal
- 4 vehicle types: Auto, Bike, Sedan, SUV
- Auto-detects traffic level and surge multiplier from time of day
- Fare breakdown showing base fare, traffic add-on, surge, and night charges
- Side-by-side layout — inputs on the left, result on the right

---

## How It Works

Fare rates were collected from Ola, Uber and Rapido across 10 Indian cities at 3 distance points each, then calibrated using least squares regression. A Random Forest model was trained on 10,000 synthetic rides generated from these real-world rates, learning how distance, time of day, traffic, surge, and vehicle type affect pricing.

**Model performance:**
| Metric | Score |
|--------|-------|
| R² Score | 97.87% |
| MAE | ₹45.50 |
| RMSE | ₹67.62 |

---

## Project Structure

```
faresense-india/
├── app.py                  # Streamlit app
├── ride_fare_model.py      # Data generation + model training
├── ride_fare_model.pkl     # Trained Random Forest model
├── ride_fare_features.pkl  # Feature column list
├── ohe_encoder.pkl         # OneHotEncoder for city & vehicle
├── city_rates.pkl          # Calibrated base rates per city
├── requirements.txt
└── README.md
```

---

## Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model (generates pkl files)
python ride_fare_model.py

# Launch the app
streamlit run app.py
```

---

## Tech Stack

- Python
- Scikit-learn — Random Forest Regressor
- Pandas, NumPy
- Streamlit

---

## Data Sources

Fare data manually collected from Ola, Uber and Rapido apps across 10 Indian cities at 3km, 7km and 12km distance points. Rates back-calculated using least squares regression.

---

*by Divya Jain*
