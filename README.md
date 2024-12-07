# Reddit Sentiment Analysis for Stock Prediction

This project analyzes Reddit discussions and sentiment related to stocks to predict stock price movements. It leverages the **VADER sentiment analyzer**, **Reddit API (via PRAW)**, and **Yahoo Finance API** for data collection and combines them with a **logistic regression model** to classify whether stock prices will rise or fall based on sentiment.

---

## Features

- Fetches Reddit posts discussing a specific stock ticker.
- Analyzes the sentiment of posts using the **VADER sentiment analysis tool**.
- Collects stock price data from Yahoo Finance.
- Merges sentiment scores with stock price data to train a machine learning model.
- Predicts future stock trends based on the sentiment of recent posts.

---

## Prerequisites

### Required Libraries

Install the following Python libraries:

```bash
pip install praw pandas numpy scikit-learn vaderSentiment yfinance tabulate
