

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import plotly.graph_objects as go

"""# Example of Stock Pulling"""

msft = yf.Ticker("MSFT")

hist = msft.history(period="max")

hist["Open"].plot(figsize=(15, 5), title="MSFT Stock Price")
plt.show()

stocks = [
    "^GSPC",
    "ETSY",
    "PINS",
    "SQ",
    "SHOP",
    "O",
    "MELI",
    "ISRG",
    "DIS",
    "BRK-B",
    "AMZN",
    "ZM",
    "PFE",
    "CLX",
    "DPZ",
    "RTX",
]

hists = {}
for s in stocks:
    tkr = yf.Ticker(s)
    history = tkr.history(period="5y")
    hists[s] = history

for stock in stocks:
    temp_df = hists[stock].copy()

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=temp_df.index,
                open=temp_df["Open"],
                high=temp_df["High"],
                low=temp_df["Low"],
                close=temp_df["Close"],
            )
        ]
    )

    fig.update_layout(
        margin=dict(l=20, r=20, t=60, b=20),
        height=300,
        paper_bgcolor="LightSteelBlue",
        title=stock,
    )

    fig.show()

import yfinance as yf
import pandas as pd
from tqdm.notebook import tqdm
from transformers import pipeline
import matplotlib.pyplot as plt

# Stock symbol to fetch news for
stock = "CLX"

# Fetching the stock data using yfinance
ticker = yf.Ticker(stock)
news = ticker.news

# Creating list to append news data
news_list = []

# Looping through the news articles and adding to the list
for i, item in tqdm(enumerate(news), total=len(news)):
    news_list.append(
        [item['title'], item['link'], item['publisher'], item['providerPublishTime']]
        # Attributes to be returned
    )

# Creating a dataframe from the news list above
news_df = pd.DataFrame(news_list, columns=["Title", "Link", "Publisher", "Datetime"])

# Convert the Datetime column from Unix timestamp to a readable format
news_df['Datetime'] = pd.to_datetime(news_df['Datetime'], unit='s')

# Step 2: Perform sentiment analysis on news titles using Hugging Face's RoBERTa model
sentiment_task = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

# Step 3: Analyze sentiment for each news title and add the results to the DataFrame
sentiment_results = news_df["Title"].apply(lambda x: sentiment_task(x)[0])
news_df["Sentiment"] = sentiment_results.apply(lambda x: x['label'])
news_df["Sentiment Score"] = sentiment_results.apply(lambda x: x['score'])

# Ensure Sentiment Score is numeric (if it's not already)
news_df["Sentiment Score"] = pd.to_numeric(news_df["Sentiment Score"], errors='coerce')

# Display the DataFrame with sentiment scores
print("\nNews with Sentiment Analysis:")
print(news_df.head())

# Step 4: Save the DataFrame to a Parquet file
news_df.to_parquet("yahoo_finance_news_with_sentiment.parquet")

# Step 5: Optional - Visualize Sentiment Scores
# Group by the 'Datetime' and calculate the mean sentiment score per day
daily_sentiment = news_df.groupby(news_df['Datetime'].dt.date)["Sentiment Score"].mean()

# Plotting the sentiment score trends over time
plt.figure(figsize=(10, 5))
daily_sentiment.plot(title="Average Sentiment Score Over Time", color='blue')
plt.xlabel("Date")
plt.ylabel("Average Sentiment Score")
plt.grid(True)
plt.show()

"""## Sentiment Analysis Prep"""

from transformers import pipeline

model = f"cardiffnlp/twitter-roberta-base-sentiment-latest"

sentiment_task = pipeline("sentiment-analysis", model=model)
sentiment_task("Market Trends for Stock Prices")

import pandas as pd

# Initialize an empty dictionary to store sentiment results
sent_results = {}
count = 0

# Loop through each row of the news DataFrame and analyze the sentiment of the news titles
for i, d in tqdm(news_df.iterrows(), total=len(news_df)):
    # Perform sentiment analysis on the news title
    sent = sentiment_task(d["Title"])

    # Store the sentiment analysis result in the dictionary with Link as the key
    sent_results[d["Link"]] = sent

    # Increment count
    count += 1

    # Stop after processing 500 news articles
    if count == 500:
        break

# Step 1: Convert sentiment results dictionary to DataFrame
sent_df = pd.DataFrame(sent_results).T

# Step 2: Extract the sentiment label and score from the sentiment result
sent_df["label"] = sent_df[0].apply(lambda x: x["label"])
sent_df["score"] = sent_df[0].apply(lambda x: x["score"])

# Drop the unnecessary 0 column
sent_df = sent_df.drop(columns=[0])

# Step 3: Merge the sentiment DataFrame with the original news DataFrame
# First, create a new DataFrame for news titles and links (from `news_df`)
news_for_merge = news_df[['Link', 'Title', 'Publisher', 'Datetime']]

# Merge sentiment data with the news DataFrame on the 'Link' column
sent_df = sent_df.merge(news_for_merge, left_index=True, right_on='Link')

# Step 4: Display the merged DataFrame
print(sent_df.head())

sent_df.groupby("label")["score"].plot(kind="hist", bins=50)
plt.legend()
plt.show()

sent_df["score_"] = sent_df["score"]

sent_df.loc[sent_df["label"] == "Negative", "score_"] = (
    sent_df.loc[sent_df["label"] == "Negative"]["score"] * -1
)

sent_df.loc[sent_df["label"] == "Neutral", "score_"] = 0

sent_df["Date"] = sent_df["Datetime"].dt.date

sent_daily = sent_df.groupby("Date")["score_"].mean()

clx_df = hists["CLX"].copy()
clx_df = clx_df.reset_index()
clx_df["Date"] = clx_df["Date"].dt.date
clx_df = clx_df.set_index("Date")

sent_and_stock = sent_daily.to_frame("sentiment").merge(
    clx_df, left_index=True, right_index=True
)

ax = sent_and_stock["sentiment"].plot(legend="Sentiment")
ax2 = ax.twinx()
sent_and_stock["Close"].plot(ax=ax2, color="orange", legend="Closing Price")
plt.show()

"""# Scrapping Stock Data from Reddit"""

!pip install praw

import yfinance as yf
import pandas as pd
import praw
import datetime
import numpy as np
import re
from transformers import pipeline, RobertaTokenizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Fetch stock data using yfinance
stock_symbol = 'AAPL'

# Define start and end dates for training and testing
start_train_date = '2024-09-01'
end_train_date = '2024-09-30'

start_test_date = '2024-10-01'
end_test_date = '2024-10-30'

# Fetch stock data for training and testing periods
train_data = yf.download(stock_symbol, start=start_train_date, end=end_train_date)
test_data = yf.download(stock_symbol, start=start_test_date, end=end_test_date)

# Check if stock data is available
if train_data.empty or test_data.empty:
    print("No data available for the specified stock during the given periods.")
else:
    print("Stock data successfully fetched for training and testing.")

# 2. Fetch Reddit data using PRAW
reddit = praw.Reddit(
    client_id="Paste your Client ID",
    client_secret="Paste Your secret code",
    user_agent="stock_sentiment_analysis"
)

subreddit_name = 'StockMarket'
start_date = datetime.date(2024, 9, 1)
end_date = datetime.date(2024, 10, 30)

subreddit = reddit.subreddit(subreddit_name)

# Fetch Reddit posts for the training period
posts = []
# Fetching a larger number of posts for training
for post in subreddit.new(limit=1000):
    post_date = datetime.datetime.utcfromtimestamp(post.created_utc).date()
    if start_date <= post_date <= end_date:
        posts.append({
            'post_id': post.id,
            'title': post.title,
            'content': post.selftext,
            'upvotes': post.score,
            'date': post_date
        })

# Check if posts are available
if posts:
    print(f"Found {len(posts)} posts between {start_date} and {end_date}.")
else:
    print("No posts found for the training period.")

# Preprocess Reddit posts and analyze sentiment
def clean_text(text):
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # Remove special characters
    text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
    text = text.lower()
    return text

# Initialize sentiment analysis model
tokenizer = RobertaTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
sentiment_task = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

# Function to analyze sentiment of long posts
def analyze_sentiment(text):
    if not text.strip():
        return 'neutral'

    chunks = [text[i:i + 512] for i in range(0, len(text), 512)]
    sentiments = [sentiment_task(chunk)[0]['label'] for chunk in chunks]
    return max(set(sentiments), key=sentiments.count)

# Clean text and analyze sentiment for each Reddit post
for post in posts:
    post['content_cleaned'] = clean_text(post['content'])
    post['sentiment'] = analyze_sentiment(post['content_cleaned'])

# Create DataFrame for Reddit data
reddit_df = pd.DataFrame(posts)

# Map sentiments to numeric values: positive = 1, negative = 0, neutral = 2
sentiment_mapping = {'positive': 1, 'negative': 0, 'neutral': 2}
reddit_df['sentiment_numeric'] = reddit_df['sentiment'].map(sentiment_mapping)

# Merge stock data with sentiment data (we can align by date)
reddit_df['date'] = pd.to_datetime(reddit_df['date'])

reddit_df

train_data['Date'] = train_data.index
test_data['Date'] = test_data.index

# Filter train and test data based on dates
train_data_filtered = train_data[(train_data['Date'] >= start_train_date) & (train_data['Date'] <= end_train_date)]
test_data_filtered = test_data[(test_data['Date'] >= start_test_date) & (test_data['Date'] <= end_test_date)]

# Sample sentiment data to match stock data dates
train_sentiment = reddit_df[reddit_df['date'].isin(train_data_filtered['Date'].dt.date)]
test_sentiment = reddit_df[reddit_df['date'].isin(test_data_filtered['Date'].dt.date)]

train_data_filtered['Date'] = pd.to_datetime(train_data_filtered['Date']).dt.date

# Ensure 'date' column in train_sentiment is in the correct format
train_sentiment['date'] = pd.to_datetime(train_sentiment['date']).dt.date

train_data_filtered

train_sentiment

"""Merging sentiment data and stock price data on date column"""

import pandas as pd

# Flatten MultiIndex columns
train_data_filtered.columns = [f'{col[0]}_{col[1]}' if col[1] else col[0] for col in train_data_filtered.columns]

# Convert dates to standard format
train_data_filtered['date'] = pd.to_datetime(train_data_filtered['Date']).dt.date
train_sentiment['date'] = pd.to_datetime(train_sentiment['date']).dt.date

# Merge with flattened columns
merged_data = pd.merge(
    train_data_filtered,
    train_sentiment,
    on='date',
    how='outer'
)

# Display merge results
# print(merged_data.info())
# print("\nSample of merged data:")
# print(merged_data.head())
merged_data = pd.DataFrame(merged_data)

merged_data.head()

"""#Data Analysis"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Descriptive Statistics
print("Descriptive Statistics:")
print(merged_data.describe())

# Missing Value Analysis
print("\nMissing Values:")
print(merged_data.isnull().sum())

# Correlation Analysis
correlation_columns = [
    'Close_AAPL', 'High_AAPL', 'Low_AAPL',
    'Open_AAPL', 'Volume_AAPL',
    'sentiment_numeric', 'upvotes'
]

correlation_matrix = merged_data[correlation_columns].corr()

plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()

# Sentiment Distribution
plt.figure(figsize=(10,6))
merged_data['sentiment'].value_counts().plot(kind='bar')
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Stock Price Trend
plt.figure(figsize=(12,6))
merged_data['Close_AAPL'].plot()
plt.title('Stock Closing Price Trend')
plt.xlabel('Sample Index')
plt.ylabel('Closing Price')
plt.tight_layout()
plt.show()

"""#Prediction model for stock price prediction"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Prepare features
merged_data['sentiment_impact'] = merged_data['sentiment_numeric'] * merged_data['upvotes']
features = ['Close_AAPL', 'High_AAPL', 'Low_AAPL', 'Open_AAPL', 'Volume_AAPL', 'sentiment_impact']

# Prepare X and y
X = merged_data[features].fillna(merged_data[features].mean())
# Predict next day's closing price
y = merged_data['Close_AAPL'].shift(-1)
# Align with shifted target
X = X[:-1]
y = y[:-1]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = rf_model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R-squared Score:", r2)

# Feature importance
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)

import matplotlib.pyplot as plt

plt.figure(figsize=(12,6))
plt.plot(y_test.values, label='Actual Prices', color='blue')
plt.plot(y_pred, label='Predicted Prices', color='red', linestyle='--')
plt.title('Actual vs Predicted Stock Prices')
plt.xlabel('Sample Index')
plt.ylabel('Stock Price')
plt.legend()
plt.tight_layout()
plt.show()

# Calculate prediction accuracy metrics
absolute_errors = np.abs(y_test.values - y_pred)
mean_absolute_error = np.mean(absolute_errors)
print(f"Mean Absolute Error: ${mean_absolute_error:.2f}")

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Comprehensive Performance Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

# Performance Report
print("Model Performance Metrics:")
print(f"Mean Absolute Error: ${mae:.2f}")
print(f"Root Mean Squared Error: ${rmse:.2f}")
print(f"R-squared Score: {r2:.4f}")
print(f"Mean Absolute Percentage Error: {mape*100:.2f}%")

# Residual Analysis
residuals = y_test.values - y_pred

plt.figure(figsize=(15,5))
plt.subplot(131)
plt.scatter(y_pred, residuals)
plt.title('Residual Plot')
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')

plt.subplot(132)
sns.histplot(residuals, kde=True)
plt.title('Residual Distribution')
plt.xlabel('Residual Value')

plt.subplot(133)
plt.plot(residuals)
plt.title('Residuals Sequence')
plt.xlabel('Sample Index')

plt.tight_layout()
plt.show()

# Feature Impact Visualization
plt.figure(figsize=(10,6))
sns.barplot(x=feature_importance['feature'], y=feature_importance['importance'])
plt.title('Feature Importance in Stock Price Prediction')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

"""#Classification Model for Stock price movement"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# Prepare target variable (stock movement)
merged_data['stock_movement'] = np.where(
    merged_data['Close_AAPL'].diff() > 0,
    1,  # Up
    np.where(merged_data['Close_AAPL'].diff() < 0, -1, 0)  # Down or Flat
)

# Prepare features
features = ['Close_AAPL', 'High_AAPL', 'Low_AAPL', 'Open_AAPL', 'Volume_AAPL',
            'sentiment_numeric', 'upvotes']

# Prepare X and y
X = merged_data[features].fillna(merged_data[features].mean())
y = merged_data['stock_movement'].shift(-1)

# Align data
X = X[:-1]
y = y[:-1]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_scaled, y_train)

# Predict and evaluate
y_pred = rf_classifier.predict(X_test_scaled)

# Performance Metrics
print("Classification Report:")
print(classification_report(y_test, y_pred))

import matplotlib.pyplot as plt
import numpy as np

# Create visualization of predicted vs actual movement
plt.figure(figsize=(15,6))
plt.plot(y_test.values, label='Actual Movement', marker='o')
plt.plot(y_pred, label='Predicted Movement', marker='x', linestyle='--')
plt.title('Actual vs Predicted Stock Movement')
plt.xlabel('Sample Index')
plt.ylabel('Movement Direction')
plt.legend()
plt.yticks([-1, 0, 1], ['Down', 'Flat', 'Up'])
plt.tight_layout()
plt.show()

# Detailed Movement Comparison
movement_comparison = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred
})
print("\nMovement Comparison:")
print(movement_comparison.value_counts())

# Accuracy for each movement type
for movement in [-1, 0, 1]:
    movement_mask = y_test.values == movement
    movement_accuracy = np.mean(y_pred[movement_mask] == y_test.values[movement_mask])
    print(f"\nAccuracy for {['Down', 'Flat', 'Up'][movement+1]} movement: {movement_accuracy:.2%}")

# Confusion Matrix Visualization
plt.figure(figsize=(8,6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix of Stock Movement Prediction')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_classifier.feature_importances_
}).sort_values('importance', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='feature', y='importance', data=feature_importance)
plt.title('Feature Importance in Stock Movement Classification')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()