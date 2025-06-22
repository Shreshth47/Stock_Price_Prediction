# 📈 Stock Price Prediction using K-Nearest Neighbors (KNN)

This project demonstrates stock price movement classification and closing price regression using K-Nearest Neighbors (KNN). We use historical stock data from **Tata Consumer Products Ltd (TATACONSUM.NS)** sourced via **Yahoo Finance** to build both classification and regression models.

## 🔍 Overview

- **Classification Task**: Predict if the stock’s closing price will go **up or down** the next day.
- **Regression Task**: Predict the **next day's closing price**.
- **Model Used**: `KNeighborsClassifier` & `KNeighborsRegressor` from `scikit-learn`
- **Parameter Tuning**: Hyperparameter tuning using `GridSearchCV`
- **Feature Engineering**: Created features based on price difference ranges:
  - `Open - Close`
  - `High - Low`

## 🛠️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- yFinance (for fetching stock data)
- Matplotlib

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/stock_price_prediction.git
   cd stock_price_prediction
    ```
2. Install the required dependencies:
   ```
   pip install pandas numpy scikit-learn matplotlib yfinance
   ```
3. Run the script:
   ```
   python stock_price.py
   ```
## Example Output
```
Training Accuracy: 0.66
Testing Accuracy: 0.61

   Actual_Close  Predicted_Close
0       742.25            734.10
1       747.00            739.30
...
```


