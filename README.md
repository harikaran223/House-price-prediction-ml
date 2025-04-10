# House Price Prediction

This project is a **web-based machine learning application** that predicts house prices in Chennai based on user inputs. The model is trained using the **Chennai House Price Prediction** dataset and deployed using **Flask** as the backend and **HTML/CSS** for the frontend interface.

## Features

- Predicts house prices based on input parameters like location, number of bedrooms, and square footage.
- Clean and simple user interface.
- Machine learning model trained using real-world data.
- Flask backend handles prediction logic and model integration.

## Tech Stack

- **Frontend**: HTML, CSS
- **Backend**: Flask (Python)
- **ML Libraries**: Pandas, NumPy, Scikit-learn, Joblib
- **Dataset**: [Chennai House Price Prediction - Kaggle](https://www.kaggle.com/datasets/)

- **Replace the house_price_model.pkl file after build the new model using pynb file.**

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/house-price-prediction.git
cd house-price-prediction

house-price-prediction/
├── templates/
│   └── index.html              # Frontend input form
│
├── static/
│   └── style.css              # (Optional) CSS styles
│
├── model/
│   └── model.pkl              # Trained ML model
│
├── app.py                     # Flask backend
├── requirements.txt           # Dependencies
├── README.md                  # Project documentation

