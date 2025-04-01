# Stock Market Predictor using LSTM

This project demonstrates the creation and deployment of a stock price prediction model using a Long Short-Term Memory (LSTM) neural network. It includes a Jupyter Notebook for training the model and a Streamlit web application for interactive visualization and prediction.

## Overview

The goal is to predict the closing price of a stock based on its historical price data. The project utilizes:
1.  **Jupyter Notebook (`Stock_Market_Prediction_Model_Creation.ipynb`):** For data fetching, preprocessing, LSTM model training, evaluation, and saving the trained model.
2.  **Streamlit App (`app.py`):** For loading the pre-trained model, allowing users to input a stock ticker, fetching data, visualizing historical trends (including Moving Averages), and displaying the predicted vs. actual closing prices for the test period.

## Features

*   **Data Fetching:** Uses `yfinance` to download historical stock data (Open, High, Low, Close, Volume, Adjusted Close).
*   **Data Visualization:**
    *   Displays raw historical stock data in a table.
    *   Plots closing prices against 50-day, 100-day, and 200-day Moving Averages (MAs).
    *   Compares original closing prices with LSTM predicted prices on the test set.
*   **LSTM Model:**
    *   Preprocesses data using `MinMaxScaler`.
    *   Uses sequences of the previous 100 days' closing prices to predict the next day's price.
    *   Employs a stacked LSTM architecture built with Keras (TensorFlow backend).
    *   Includes Dropout layers to prevent overfitting.
*   **Interactive Web Application:**
    *   Built with Streamlit for a user-friendly interface.
    *   Allows users to enter any valid stock ticker symbol.
    *   Dynamically updates data and plots based on user input.
    *   Includes basic error handling for invalid tickers or data fetching issues.

## Technologies Used

*   **Programming Language:** Python 3
*   **Data Handling:** Pandas, NumPy
*   **Data Fetching:** yfinance
*   **Machine Learning:** TensorFlow (Keras API - Sequential, LSTM, Dense, Dropout), Scikit-learn (MinMaxScaler)
*   **Visualization:** Matplotlib
*   **Web Framework:** Streamlit

## Project Structure

## Setup and Installation

1.  **Prerequisites:**
    *   Python 3.7+
    *   pip (Python package installer)

2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```

3.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    ```
    *   On Windows: `venv\Scripts\activate`
    *   On macOS/Linux: `source venv/bin/activate`

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: TensorFlow can be a large download.)*

## Usage

1.  **Run the Streamlit Web Application:**
    *   Make sure your virtual environment is activated.
    *   Execute the following command in your terminal:
        ```bash
        streamlit run app.py
        ```
    *   Open your web browser and navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).
    *   Enter a valid stock ticker symbol (e.g., `GOOGL`, `AAPL`, `MSFT`) in the text input box and press Enter.
    *   The application will display the data, MA plots, and the prediction vs. original price plot.

2.  **Retraining the Model (Optional):**
    *   The repository includes a pre-trained model (`Stock Predictions Model.keras`).
    *   If you wish to retrain the model (e.g., with different parameters or data), open and run the `Stock_Market_Prediction_Model_Creation.ipynb` notebook using Jupyter Lab or Jupyter Notebook.
    *   Running the notebook completely will overwrite the existing `.keras` file with the newly trained model.

## How It Works

1.  **Data Collection:** Historical stock data is fetched using `yfinance`.
2.  **Preprocessing:** The 'Close' prices are extracted and scaled to a range between 0 and 1 using `MinMaxScaler`.
3.  **Sequence Creation:** The scaled data is transformed into sequences where each input sequence consists of 100 previous closing prices, and the target is the next day's closing price.
4.  **LSTM Model:** A stacked LSTM network learns temporal dependencies from these sequences. Dropout layers help regularize the model.
5.  **Training (Notebook):** The model is trained on 80% of the historical data.
6.  **Prediction (App):** The pre-trained model is loaded. For the selected stock, the last 20% of data (plus the preceding 100 days for sequence creation) is used as the test set. Predictions are made on this test set.
7.  **Evaluation/Visualization:** Predictions are inverse-scaled back to the original price range and plotted against the actual closing prices for visual comparison. Moving average plots help visualize trends.

## Future Improvements

*   Incorporate more features (e.g., Volume, Technical Indicators) for potentially better predictions.
*   Implement more advanced time-series models (e.g., Attention mechanisms, Transformers).
*   Add quantitative evaluation metrics (RMSE, MAE) to the Streamlit app.
*   Deploy the Streamlit application to a cloud platform (e.g., Streamlit Community Cloud, Heroku, AWS).
*   Implement more sophisticated error handling and user feedback.
*   Allow users to select date ranges.
