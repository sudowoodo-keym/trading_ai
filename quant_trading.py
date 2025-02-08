import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

class QuickAnalysis:
    def main(ticker: str, setPeriod: str, setInterval: str):
        # Download historical data
        df = yf.download(ticker, period=setPeriod, interval=setInterval)
        
        # Display first few rows
        print(df.head())

        # Plot closing price
        plt.figure(figsize=(12,5))
        plt.plot(df.index, df['Close'], label='Closing Price', color='blue')

        # Formatting
        plt.title(f"{ticker} Price Over Time")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.grid()
        plt.show()
        # Best for quick trend analysis

class InteractiveAnalysis:
    def main(ticker: str, setPeriod: str, setInterval: str):

        short_window=7 
        long_window=30

        symbol = yf.Ticker(ticker)
        df = symbol.history(period=setPeriod, interval=setInterval)

        print(df.head())

        # Calculate Simple Moving Averages (SMA)
        df[f"SMA_{short_window}"] = df["Close"].rolling(window=short_window).mean()
        df[f"SMA_{long_window}"] = df["Close"].rolling(window=long_window).mean()

        # Calculate Exponential Moving Averages (EMA)
        df[f"EMA_{short_window}"] = df["Close"].ewm(span=short_window, adjust=False).mean()
        df[f"EMA_{long_window}"] = df["Close"].ewm(span=long_window, adjust=False).mean()

        # Calculate RSI (Relative Strength Index)
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

        # Calculate MACD (12-period EMA minus 26-period EMA)
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()  # 12-period EMA
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()  # 26-period EMA
        df['MACD'] = df['EMA_12'] - df['EMA_26']  # MACD line
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()  # Signal Line (9-period EMA of MACD)
        df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']  # Histogram (difference between MACD and Signal Line)

        # Create subplots (2 rows: Candlestick + RSI)
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                            row_heights=[0.6, 0.2, 0.2], 
                            subplot_titles=(f"{ticker} Candlestick & Moving Averages", "Relative Strength Index (RSI)"))
        
        # Add candlestick
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'], 
            high=df['High'],
            low=df['Low'], 
            close=df['Close'],
            name="Candlestick",
            increasing_line_color = 'green',
            decreasing_line_color = 'red'
        ), row=1, col=1)

        # Add Moving Averages (SMA)
        fig.add_trace(go.Scatter(x=df.index, y=df[f"SMA_{short_window}"], mode='lines', name=f"SMA {short_window}", line=dict(color="blue", width=1.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df[f"SMA_{long_window}"], mode='lines', name=f"SMA {long_window}", line=dict(color="orange", width=1.5)), row=1, col=1)
        # Add Moving Averages (EMA)
        fig.add_trace(go.Scatter(x=df.index, y=df[f"EMA_{short_window}"], mode='lines', name=f"EMA {short_window}", line=dict(color="purple", width=1, dash="dot")), row=1, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df[f"EMA_{long_window}"], mode='lines', name=f"EMA {long_window}", line=dict(color="pink", width=1, dash="dot")), row=1, col=1)

        # Add MACD (Histogram)
        fig.add_trace(go.Bar(x=df.index, y=df['MACD_Histogram'], name='MACD Histogram', marker=dict(color=df['MACD_Histogram'].apply(lambda x: 'green' if x >= 0 else 'red'))), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD', line=dict(color="blue")), row=2, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['Signal_Line'], mode='lines', name='Signal Line', line=dict(color="orange", dash="dot")), row=2, col=1)

        # Add RSI Indicator (Separate Plot)
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], mode='lines', name="RSI", line=dict(color="black", width=1.5)), row=2, col=1)

        # Add Volume as a Bar Chart
        fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name="Volume", marker=dict(color="magenta", opacity=0.4)), row=3, col=1)

        # Add RSI Overbought/Oversold Levels
        fig.add_hline(y=70, line=dict(color="red", dash="dash"), row=2, col=1)
        fig.add_hline(y=30, line=dict(color="green", dash="dash"), row=2, col=1)

        # Add title & layout
        fig.update_layout(
            xaxis_rangeslider_visible=False,  # Hide range slider
            height=800,  # Increase figure height
            title_text=f"{ticker} Candlestick, Moving Averages, RSI, & Volume"
        )

        # Show interactive chart
        fig.show()

        X, y = InteractiveAnalysis.prepare(df, short_window, long_window)
        InteractiveAnalysis.training(ticker, X, y)

    def prepare(df, short_window, long_window, lookback=10):

        SMA7 = f"SMA_{short_window}"
        SMA30 = f"SMA_{long_window}"
        EMA7 = f"EMA_{short_window}"
        EMA30 = f"EMA_{long_window}"

        # Calculate Price Change
        df['Price_Change'] = df['Close'].pct_change() 

        df[f"SMA_{short_window}"] = df["Close"].rolling(window=short_window).mean()
        df[f"SMA_{long_window}"] = df["Close"].rolling(window=long_window).mean()

        # Calculate Exponential Moving Averages (EMA)
        df[f"EMA_{short_window}"] = df["Close"].ewm(span=short_window, adjust=False).mean()
        df[f"EMA_{long_window}"] = df["Close"].ewm(span=long_window, adjust=False).mean()

        # Bollinger Bands
        df["Bollinger_Upper"] = df["Close"].rolling(20).mean() + (df["Close"].rolling(20).std() * 2)
        df["Bollinger_Lower"] = df["Close"].rolling(20).mean() - (df["Close"].rolling(20).std() * 2)

        # Calculate RSI (Relative Strength Index)
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

        # Define features (add more as needed)
        features = ['Close', 'RSI', f"SMA_{short_window}", f"SMA_{long_window}"]

        # Define the target: 1 if price increases tomorrow, else 0
        df["Target"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)

        df = df.dropna()

        # print("Columns in DataFrame:", df.columns)
        # print(df[['Close', 'Price_Change', f"SMA_{short_window}", f"SMA_{long_window}", f"EMA_{short_window}", f"EMA_{long_window}", "RSI"]].tail())

        # Scale the features using StandardScaler
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df[features])

        # Prepare the input (X) and target (y) data
        X, y = [], []
        for i in range (lookback, len(scaled_data)-1):
            X.append(scaled_data[i-lookback:i])
            y.append(df['Target'].iloc[i])

        X = np.array(X)
        X = X.reshape(X.shape[0], -1)
        # Return the prepared data for model training
        return X, np.array(y)
    
    def training(ticker, X, y):
        # Split the data into training and testing sets (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Initialize RandomForestClassifier
        model = RandomForestClassifier(random_state=42)

        # Define hyperparameters to tune
        param_grid = {
            'n_estimators': [50, 75, 100], # Reduce total trees for faster training
            'max_depth': [10, 15, 30], # Limit tree depth to prevent excessive memory usage
            'min_samples_split': [5, 10], # Prevent very deep splits
            'min_samples_leaf': [2, 4], # Ensures each leaf has enough data points
            'max_features': ['sqrt'] # Limit feature selection per split
        }

        # Perform GridSearchCV
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
        # Train the model
        grid_search.fit(X_train, y_train)

        # Best parameters and model
        print("Best Hyperparameters:", grid_search.best_params_)
        best_model = grid_search.best_estimator_

        # Evaluate the best model
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{ticker}\nImproved Accuracy: {accuracy:.2f}")

        # Ensure it's a 1D array
        y_pred = np.array(y_pred).flatten()

        # Count predictions
        up_predictions = np.sum(y_pred == 1)  # Count occurrences of 1 (price up)
        down_predictions = np.sum(y_pred == 0)  # Count occurrences of 0 (price down)

        # Calculate percentages
        total_predictions = len(y_pred)
        up_percentage = (up_predictions / total_predictions) * 100
        down_percentage = (down_predictions / total_predictions) * 100

        # Print results
        print(f"🔼 Predicted UP: {up_percentage:.2f}% of the time")
        print(f"🔽 Predicted DOWN: {down_percentage:.2f}% of the time")

if __name__ == "__main__":
    ticker = input("Enter ticker/symbol from Yahoo Finance:\n")
    period = input("""Enter time frame:
                    1d	    Last 1 day
                    5d	    Last 5 days
                    1mo	    Last 1 month
                    3mo	    Last 3 months
                    6mo	    Last 6 months
                    1y	    Last 1 year
                    2y	    Last 2 years
                    5y	    Last 5 years
                    10y	    Last 10 years
                    ytd	    Year to date (from Jan 1 of current year)
                    max	    All available data (since asset listing)\n""")
    interval = input("""Enter interval/frequency:
                    1m	    1-minute interval (only for recent 7 days)
                    2m	    2-minute interval (last 60 days)
                    5m	    5-minute interval (last 60 days)
                    15m	    15-minute interval (last 60 days)
                    30m	    30-minute interval (last 60 days)
                    1h	    1-hour interval (last 730 days ~ 2 years)
                    1d	    1-day interval (works for all periods)
                    5d	    5-day interval
                    1wk	    1-week interval
                    1mo	    1-month interval
                    3mo	    3-month interval\n""")
    setting = int(input("""Enter setting:
    1. Quick Analysis
    2. Interactive Analysis w/ AI Prediction
    CTRL-C to kill program.\n"""))
    match setting:
        case 1:
            QuickAnalysis.main(ticker, period, interval)
        case 2:
            InteractiveAnalysis.main(ticker, period, interval)
        case _:
            print("Invalid option")