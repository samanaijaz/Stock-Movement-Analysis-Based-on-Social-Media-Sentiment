# 📈 Stock Movement Analysis Based on Social Media Sentiment

This project predicts stock prices and movements by combining historical data with social media sentiment analysis. It leverages machine learning techniques to analyze trends and sentiment from social media platforms to provide more informed stock predictions.

---

## 🚀 Features

- Fetches real-time stock price data using **Yahoo Finance API**.
- Performs sentiment analysis on social media data (e.g., Reddit, Twitter).
- Combines historical trends with sentiment-driven insights for enhanced predictions.
- Visualizes trends and predictions with interactive graphs.

---

## 🔧 Installation and Setup

### Prerequisites

- **Python 3.7+**
- **pip** package manager
- Access to Reddit API keys for data scraping and sentiment analysis.

### Steps to Set Up

1. Clone the repository:
   ```bash
   git clone https://github.com/samanaijaz/Stock-Movement-Analysis-Based-on-Social-Media-Sentiment.git
   cd Stock-Movement-Analysis-Based-on-Social-Media-Sentiment

2. Install required dependencies:
   ```bash
    pip install -r requirements.txt

3. Add your API keys for sentiment analysis:
    ```bash
    REDDIT_API_KEY=<your_reddit_api_key>
    REDDIT_API_SECRET=<your_reddit_api_secret>

4. Run the application:
    ```bash
    python stock movement analysis.py

---


### 🛠️ Tools and Technologies
- Python for backend processing and machine learning.
- Yahoo Finance API to fetch stock market data.
- Natural Language Toolkit (NLTK) and TextBlob for sentiment analysis.
- Pandas and NumPy for data manipulation.
- Matplotlib and Plotly for data visualization.

---

### 📊 Results
- Stock Trend Analysis: Visualized historical trends using Yahoo Finance and Reddit data.
- Sentiment Analysis: Extracted and analyzed social media sentiment for impact on stock prices.
- Integrated Prediction Model: Improved stock forecasting accuracy by combining sentiment scores with historical trends.
