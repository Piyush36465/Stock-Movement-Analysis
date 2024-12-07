import praw
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import yfinance as yf
from tabulate import tabulate

# Set up Reddit API credentials
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = "Reddit News Sentiment Analysis"

# Initialize Reddit API
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT
)

# Fetch news and discussions from Reddit
def fetch_reddit_posts(subreddit_name, query, limit=100):
    subreddit = reddit.subreddit(subreddit_name)
    posts = []
    for submission in subreddit.search(query, limit=limit):
        human_readable_time = pd.to_datetime(submission.created_utc, unit='s')  # Convert to datetime
        posts.append([submission.title + " " + submission.selftext, human_readable_time])
    return pd.DataFrame(posts, columns=["Text", "Timestamp"])

# Example: Fetch posts mentioning "AAPL" from 'stocks' subreddit
reddit_df = fetch_reddit_posts("stocks", "AAPL", limit=200)
print("\nFetched Reddit Posts:")
print(tabulate(reddit_df.head(), headers="keys", tablefmt="pretty"))

# Initialize VADER
analyzer = SentimentIntensityAnalyzer()

# Add sentiment scores
def analyze_sentiment(df):
    df["Sentiment"] = df["Text"].apply(lambda x: analyzer.polarity_scores(x)["compound"])
    return df

reddit_df = analyze_sentiment(reddit_df)
print("\nReddit Posts with Sentiment Scores:")
print(tabulate(reddit_df.head(), headers="keys", tablefmt="pretty"))

# Fetch stock data and ensure stock_data has a flat index and "Date" column
def fetch_stock_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data.reset_index(inplace=True)  # Reset index to make Date a column
    if isinstance(stock_data.columns, pd.MultiIndex):
        stock_data.columns = stock_data.columns.get_level_values(0)  # Flatten multi-level index
    stock_data["Date"] = pd.to_datetime(stock_data["Date"]).dt.date  # Ensure Date column is in date format
    return stock_data[["Date", "Close"]]  # Return relevant columns

# Example: Get historical data for AAPL
stock_data = fetch_stock_data("AAPL", "2023-01-01", "2023-01-31")
print("\nFetched Stock Data:")
print(tabulate(stock_data.head(), headers="keys", tablefmt="pretty"))

# Ensure reddit_df["Date"] matches the format
reddit_df["Date"] = pd.to_datetime(reddit_df["Timestamp"]).dt.date

# Check the structure of the columns before merging
print("Reddit DataFrame columns:", reddit_df.columns)
print("Stock DataFrame columns:", stock_data.columns)

# Merge DataFrames on Date
merged_df = pd.merge(reddit_df, stock_data, on="Date", how="inner")
print("\nMerged Reddit and Stock Data:")
print(tabulate(merged_df.head(), headers="keys", tablefmt="pretty"))

# Feature Engineering
merged_df["Label"] = (merged_df["Close"].shift(-1) > merged_df["Close"]).astype(int)  # 1 if price goes up
merged_df = merged_df.dropna()
print(merged_df.head())

# Check the number of samples in merged_df
print(f"Number of samples in merged_df: {len(merged_df)}")

if len(merged_df) < 2:
    print("Not enough data to perform train-test split. Adjust the query or date range.")
else:
    # Prepare features and labels
    X = merged_df[["Sentiment"]]
    y = merged_df["Label"]

    # Split into train-test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train logistic regression
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    print("\nModel Accuracy:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

    # Predict sentiment-driven stock trends
    latest_posts = fetch_reddit_posts("stocks", "AAPL", limit=50)
    latest_posts = analyze_sentiment(latest_posts)
    latest_posts["Prediction"] = model.predict(latest_posts[["Sentiment"]])
    print("\nLatest Posts with Sentiment Predictions:")
    print(tabulate(latest_posts[["Text", "Timestamp", "Sentiment", "Prediction"]].head(), headers="keys", tablefmt="pretty"))

