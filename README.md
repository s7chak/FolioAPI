# Folio API

## Description

Folio API is the backend service for the Folio app, providing comprehensive portfolio analysis as an API. It calculates various metrics and analyzes data to offer insights into the financial health of your stock portfolio.

## Features

### Current Metrics ('facts')

- **Total Cost:** The total cost of your portfolio.
- **Total Worth:** The total worth of your portfolio.
- **Change:** The absolute change in the portfolio's value.
- **Change %:** The percentage change in the portfolio's value.

### Historical Metrics

- **Weighted Monthly Return:** The average monthly return considering the stock weights.
- **Weighted Daily Return:** The average daily return considering the stock weights.
- **Weighted CAGR:** The Compound Annual Growth Rate (CAGR) considering the stock weights.
- **Weighted Sharpe Ratio:** A measure of risk-adjusted return considering the stock weights.
- **Weighted Volatility:** The level of variation in portfolio returns considering the stock weights.

### Correlations

- Calculate correlations between each stock in the portfolio.

### Anomalies

- Identify anomalies in a chosen list of stocks over a specified historical period.

### Optimal Ratio Calculation

- Calculate the optimal ratio for the highest return and lowest risk portfolio among chosen stocks.

### Sentiment Analysis

- Analyze headlines from various RSS feeds to derive a sentiment score for headlines containing portfolio stocks.

## Setup

1. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the Flask app:

   ```bash
   python app.py
   ```

3. Access the API at [http://localhost:8091](http://localhost:8091).

## Dependencies

- flask
- flask-session
- feedparser
- pandas
- plotly
- yfinance
- tsmoothie
- matplotlib
- scikit-learn
- cvxpy
- nltk

## Usage

- Visit [http://localhost:8091](http://localhost:8091) to access the Folio API locally.


## Docker Building Steps:


### Step 1: Build the Docker Image using DockerFile

Open a terminal and navigate to the directory containing the Dockerfile. Run the following command to build the Docker image:

```bash
docker build -t folio-api .
```

### Step 2: Run the Docker Container

Once the image is built, you can run a Docker container using the following command:

```bash
docker run -p 8080:80 folio-api
```

This command maps port 8080 on your host machine to port 80 in the Docker container. Adjust the ports as needed based on your requirements.

### Step 3: Access the Folio API

Visit [http://localhost:8080](http://localhost:8080) in your web browser to access the Folio API running inside the Docker container.
The container is deployed to Google Cloud Run for the production backend of this app.


