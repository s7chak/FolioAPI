import base64
import math
from collections import OrderedDict
from io import BytesIO

import cvxpy as cp
import feedparser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
from flask import session
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import yfinance as yf
from ops import Config


class DataOperator():

    def __init__(self):
        pass

class ProcessOperator:
    def __init__(self, portfolio_returns, benchmark_returns):
        self.portfolio_returns = portfolio_returns
        self.benchmark_returns = benchmark_returns

    def calculate_stddev(self):
        portfolio_stddev = self.portfolio_returns.std()
        return portfolio_stddev

    def calculate_var(self, confidence_level=0.95):
        var = np.percentile(self.portfolio_returns, 100 * (1 - confidence_level))
        return var

    def calculate_total_return(self):
        total_return = (1 + self.portfolio_returns).prod() - 1
        return total_return

    def calculate_alpha_beta(self):
        cov_matrix = np.cov(self.portfolio_returns, self.benchmark_returns)
        portfolio_stddev = self.portfolio_returns.std()
        benchmark_stddev = self.benchmark_returns.std()
        beta = cov_matrix[0, 1] / benchmark_stddev ** 2
        alpha = (self.calculate_total_return() - 0.02 - beta * (self.benchmark_returns.mean() - 0.02))
        return alpha, beta

    def calculate_treynor_ratio(self, risk_free_rate=0.02):
        beta = self.calculate_alpha_beta()[1]
        excess_return = self.calculate_total_return() - risk_free_rate
        treynor_ratio = excess_return / beta
        return treynor_ratio

    def calculate_max_drawdown(self):
        cumulative_returns = (1 + self.portfolio_returns).cumprod()
        max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()
        return max_drawdown

    def calculate_calmar_ratio(self):
        cagr = (1 + self.portfolio_returns).prod() ** (252 / len(self.portfolio_returns)) - 1
        max_drawdown = self.calculate_max_drawdown()
        calmar_ratio = cagr / abs(max_drawdown)
        return calmar_ratio

    def calculate_stats(self):
        folio_stats = {
            'Portfolio Standard Deviation': f'{self.calculate_stddev():.3f}',
            'Portfolio VaR (95%)': f'{self.calculate_var(0.95):.3f}%',
            'Portfolio Total Return': f'{self.calculate_total_return():.3f}%',
            'Portfolio Alpha': f'{self.calculate_alpha_beta()[0]:.3f}',
            'Portfolio Beta': f'{self.calculate_alpha_beta()[1]:.3f}',
            'Treynor Ratio': f'{self.calculate_treynor_ratio():.3f}',
            'Max Drawdown': f'{self.calculate_max_drawdown():.3f}%',
            'Calmar Ratio': f'{self.calculate_calmar_ratio():.3f}',
        }
        for key, value in folio_stats.items():
            if (isinstance(value, float) and math.isnan(value)) or 'nan' in value:
                folio_stats[key] = 'N/A'

        return folio_stats



class NewsOperator():

    def calculate_symbol_sentiment(self, headlines_df):
        symbol_sentiment = {}

        for index, row in headlines_df.iterrows():
            symbols = [symbol.strip() for symbol in row['Stocks'].split(',')]
            sentiment_score = row['Sentiment']
            for symbol in symbols:
                if symbol not in symbol_sentiment:
                    symbol_sentiment[symbol] = {
                        'TotalScore': 0.0,
                        'Count': 0
                    }
                symbol_sentiment[symbol]['TotalScore'] += sentiment_score
                symbol_sentiment[symbol]['Count'] += 1

        for symbol, data in symbol_sentiment.items():
            data['AverageScore'] = round(data['TotalScore'] / data['Count'],3)
        result = dict(sorted(symbol_sentiment.items(), key=lambda item: item[1]['AverageScore'], reverse=True))
        return result

    def calculate_sentiment_scores(self, headlines_df):
        sia = SentimentIntensityAnalyzer()
        headlines_df['Sentiment'] = 0.0
        for index, row in headlines_df.iterrows():
            text = f"{row['Headline']}. {row['Summary']}"
            sentiment_score = sia.polarity_scores(text)['compound']
            headlines_df.at[index, 'Sentiment'] = sentiment_score

        scores = self.calculate_symbol_sentiment(headlines_df)
        return scores

    def get_stock_synonyms(self, stocks):
        res = {}
        for s in stocks:
            res[s] = Config.stock_company_map[s] if s in Config.stock_company_map else [s]
        return res

    def headline_analysis(self, code, stocks):
        if code not in session:
            session[code] = {}
        if 'headlines' in session[code] and session[code]['headlines']['total']!=0:
            return session[code]['headlines']

        stock_names = [s['name'] for s in stocks]
        company_names = self.get_stock_synonyms(stock_names)
        rss_urls = ["https://seekingalpha.com/market_currents.xml",
                    "https://seekingalpha.com/tag/long-ideas.xml",
                    "http://feeds.marketwatch.com/marketwatch/topstories/",
                    "http://feeds.marketwatch.com/marketwatch/marketpulse/",
                    "http://feeds.marketwatch.com/marketwatch/realtimeheadlines/",
                    "http://feeds.marketwatch.com/marketwatch/StocksToWatch/",
                    "https://www.fool.com/a/feeds/foolwatch?format=rss2&id=foolwatch&apikey=foolwatch-feed",
                    "https://www.investing.com/rss/news.rss",
                    "https://www.moneycontrol.com/rss/latestnews.xml",
                    "https://ragingbull.com/feed/",
                    "https://stocksnewsfeed.com/feed/",
                    ]
        headline_data = self.fetch_headlines(company_names, rss_urls)
        result = {'message':''}
        if not headline_data.empty:
            sent_scores = self.calculate_sentiment_scores(headline_data)
            result['Sentiment'] = {x:sent_scores[x]['AverageScore'] for x in sent_scores} #sent_scores
            result['total'] = headline_data.shape[0]
            result['message'] = str(headline_data.shape[0]) + " articles analyzed."
        else:
            result['total'] = 0
            result['message'] = "No relevant news found in rss."
        session[code]['headlines'] = result
        return result


    def fetch_headlines(self, stock_synonyms, rss_urls, max_articles=200):
        headlines_df = pd.DataFrame(columns=['Stocks', 'Headline', 'Link', 'Summary'])
        synonym_to_symbol = {syn.lower(): symbol for symbol, synonyms in stock_synonyms.items() for syn in synonyms}

        for rss_url in rss_urls:
            feed = feedparser.parse(rss_url)
            for entry in feed.entries[:max_articles]:
                headline, link, date, summary = entry.title, entry.link, entry.published, entry.summary if 'summary' in entry else ''
                mentioned_stocks = [synonym_to_symbol.get(word.lower(), word.lower()) for word in
                                    headline.split() + summary.split()]
                mentioned_stocks = [stock for stock in mentioned_stocks if stock in stock_synonyms]

                if mentioned_stocks:
                    new_data = {'Date':date,'Stocks': ', '.join(list(set(mentioned_stocks))), 'Headline': headline, 'Link': link,
                                'Summary': summary}
                    new_df = pd.DataFrame([new_data])
                    headlines_df = pd.concat([headlines_df, new_df], ignore_index=True)

        return headlines_df


class OptimalOperator():

    def calculate_optimal_frontier_cvx(self, selected_stocks, num_points=100):
        # historical_data = session[code]['historical']
        num_assets = len(selected_stocks)
        mu = ...  # Calculate expected returns for selected stocks
        sigma = ...  # Calculate covariance matrix for selected stocks
        weights = np.linspace(0, 1, num_points)
        optimal_portfolio_images = []

        for weight in weights:
            x = cp.Parameter(num_assets, nonneg=True)
            x.value = np.ones(num_assets) * weight
            objective = cp.Minimize(cp.quad_form(x, sigma))
            constraints = [cp.sum(x) == 1]
            problem = cp.Problem(objective, constraints)
            problem.solve()
            optimal_weights = x.value
            # optimal_portfolio_image = generate_optimal_portfolio_image(optimal_weights, mu, sigma)
            # optimal_portfolio_images.append(optimal_portfolio_image)

        return None

    def calculate_optimal_frontier(self, metadata, returns, selected_stocks, num_points=1000):
        np.random.seed(42)
        num_assets = len(selected_stocks)
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        p_weights = []
        p_ret = []
        p_vol = []

        for _ in range(num_points):
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            p_weights.append(weights)

            returns = np.sum(weights * mean_returns)
            p_ret.append(returns)

            var = np.dot(weights.T, np.dot(cov_matrix, weights))
            sd = np.sqrt(var)
            ann_sd = sd * np.sqrt(250)
            p_vol.append(ann_sd)

        data = {'Returns': p_ret, 'Volatility': p_vol}

        for i, symbol in enumerate(selected_stocks):
            data[symbol + ' weight'] = [w[i] for w in p_weights]

        portfolios = pd.DataFrame(data)
        sharpe_ratios = portfolios['Returns'] / portfolios['Volatility']
        max_sharpe_idx = sharpe_ratios.idxmax()
        max_sharpe_portfolio = portfolios.iloc[max_sharpe_idx]
        max_sharpe_df = pd.DataFrame(max_sharpe_portfolio).transpose()
        max_sharpe_df['Label'] = 'Max Sharpe Ratio'
        max_sharpe_df['Color'] = 'green'

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(x=portfolios['Volatility'], y=portfolios['Returns'],
                       mode='markers',
                       marker=dict(size=7, color=sharpe_ratios, colorscale='RdBu',
                                   colorbar=dict(title='Sharpe Ratio')))
        )

        fig.add_trace(
            go.Scatter(x=max_sharpe_df['Volatility'], y=max_sharpe_df['Returns'],
                       mode='markers+text',
                       marker=dict(size=10, color='green'),
                       text=['Max Sharpe Ratio'],
                       textposition='top right')
        )

        fig.update_layout(
            title='Efficient Frontier with Max Sharpe Portfolio',
            xaxis_title='Volatility',
            yaxis_title='Returns',
            coloraxis_colorbar=dict(title='Sharpe Ratio')
        )
        optimal_weights = p_weights[max_sharpe_idx].tolist()
        result = []
        for i, symbol in enumerate(selected_stocks):
            weight = optimal_weights[i]
            percentage = metadata[i]['percentage']

            result.append({
                'name': symbol,
                'percentage': percentage,
                'weight': weight
            })
        total_weight = sum(entry['weight'] for entry in result)
        for entry in result:
            entry['weight'] /= total_weight
        for entry in result:
            entry['weight'] = round(entry['weight'] * 100, 2)

        result = sorted(result, key=lambda x: x['weight'], reverse=True)
        img_bytes = pio.to_image(fig, format="png")
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')

        return {'optimal_weights': result, 'efficient_frontier_plot': img_base64}



class StockDeepDiver():
    def __init__(self):
        self.tickers = None
        self.ticker = None

    def get_company_info(self, symbol):
        try:
            info = self.ticker.info
            if info:
                company_info = {
                    "Company Name": info.get("longName", ""),
                    "Industry": info.get("industry", ""),
                    "Exchange": info.get("exchange", ""),
                    "Sector": info.get("sector", ""),
                    "Country": info.get("country", "")
                }
                return company_info
            return None
        except Exception as e:
            print(f"Error occurred while fetching company info for symbol {symbol}: {str(e)}")
            return None

    def get_financial_metrics(self, symbol):
        try:
            info = self.ticker.info
            financial_metrics = {
                "Market Cap": info.get("marketCap", ""),
                "Enterprise Value": info.get("enterpriseValue", ""),
                "Revenue": info.get("revenue", ""),
                "Earnings": info.get("earnings", ""),
                "EV/EBITDA": info.get("enterpriseToEbitda", ""),
                "Debt to Equity Ratio": info.get("debtToEquity", ""),
                "Earnings per Share (EPS)": info.get("forwardEps", ""),
                "P/E Ratio": info.get("trailingPE", ""),
                "Forward P/E": info.get("forwardPE", ""),
                "P/S Ratio": info.get("priceToSalesTrailing12Months", ""),
                "P/B Ratio": info.get("priceToBook", ""),
                "Price to Earnings Growth (PEG) Ratio": info.get("pegRatio", ""),
                "Dividend Yield": info.get("dividendYield", ""),
                "Current Ratio": info.get("currentRatio", ""),
                "Return on Equity (ROE)": info.get("returnOnEquity", ""),
                "Return on Assets (ROA)": info.get("returnOnAssets", ""),
                "Gross Margin": info.get("grossMargins", ""),
                "Operating Margin": info.get("operatingMargins", ""),
                "Net Margin": info.get("profitMargins", ""),
                "Volatility": info.get("volatility", ""),
                "52-Week High/Low": f"{info.get('fiftyTwoWeekHigh', '')} / {info.get('fiftyTwoWeekLow', '')}",
                "Average Volume": info.get("averageVolume", "")
            }
            return financial_metrics
        except Exception as e:
            print(f"Error occurred while fetching financial metrics for symbol {self.ticker.info['symbol']}: {str(e)}")
            return None

    def get_stock_metrics(self, symbol):
        try:
            stock = yf.Ticker(symbol)
            self.ticker = stock
            # Get basic company information
            company_info = self.get_company_info(symbol)
            # Get financial metrics
            financials = self.get_financial_metrics(symbol)
            result = OrderedDict()
            result.update(company_info)
            result.update(financials)
            return result
        except Exception as e:
            print(f"Error occurred: {e}")
            return {}

    def plot_stock_vs_sp500(self, symbol, start):
        try:
            stock_data = yf.download(symbol, start=start)
            sp500_data = yf.download('^GSPC', start=start)
            normalized_stock_data = stock_data['Adj Close'] / stock_data['Adj Close'].iloc[0]
            normalized_sp500_data = sp500_data['Adj Close'] / sp500_data['Adj Close'].iloc[0]

            plt.figure(figsize=(10, 6))
            plt.plot(normalized_stock_data, label=f'{symbol} Normalized')
            plt.plot(normalized_sp500_data, label='SP500 Normalized')
            plt.title(f'{symbol} vs S&P 500')
            plt.xlabel('Date')
            plt.ylabel('Normalized Price')
            plt.legend()
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
            return img_base64
        except Exception as e:
            print(f"Error occurred: {e}")
            return ''

    def plot_stock(self, symbol, start):
        try:
            stock_data = yf.download(symbol, start=start)
            normalized_stock_data = stock_data['Adj Close'] / stock_data['Adj Close'].iloc[0]
            plt.figure(figsize=(10, 6))
            plt.plot(normalized_stock_data, label=f'{symbol}')
            plt.title(f'{symbol}')
            plt.xlabel('Date')
            plt.ylabel('Price')
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
            return img_base64
        except Exception as e:
            print(f"Error occurred: {e}")
            return ''

    def score_stocks(self):
        fund_df = pd.DataFrame()
        for ticker in self.tickers:
            t = pd.DataFrame(ticker.info)[:1]
            fund_df = pd.concat([fund_df, t])
        data = fund_df[Config.analysis_fields]
        uvs = self.uvs_scoring(data)
        ufs = self.ufs_scoring(uvs)
        final = self.all_scoring(ufs)
        final['ThresholdScore'] = round((final['UVS'] + final['UFS']) / 20, 2)
        final.sort_values('10Score', ascending=False)
        res = final.T.to_dict('dict')
        return res

    # Fundamental Score
    def ufs_scoring(self, d):
        # UFS = (normalized revenue growth score + normalized earnings growth score + normalized profit margin score + normalized ROE score + normalized debt to equity ratio score + normalized PEG ratio score) / 6
        # w1 = 20, w2 = 20, w3 = 10, w4 = 20, w5 = 15, w6 = 15
        vars = Config.ufs_needs + Config.uvs_needs + [Config.eveb]
        d = d[vars + ['UVS']]
        my_thresholds = Config.ufs_thresholds
        for m in my_thresholds.keys():
            d[m + '_score'] = round((d[m].values[0] - my_thresholds[m][0]) / (my_thresholds[m][1] - my_thresholds[m][0]), 2)
            d[m + '_score'].fillna(0, inplace=True)

        d['UFS'] = round((Config.ufs_weights[Config.rg] * d[Config.rg + '_score'] + Config.ufs_weights[Config.eg] * d[Config.eg + '_score'] + Config.ufs_weights[Config.pm] * d[Config.pm + '_score'] + Config.ufs_weights[
                              Config.roe] * d[Config.roe + '_score'] + Config.ufs_weights[Config.dte] * d[Config.dte + '_score'] + Config.ufs_weights[Config.peg] * d[Config.peg + '_score']) / 100, 2)
        return d.sort_values('UFS', ascending=False)

    def all_scoring(self, d):
        # UFS = (normalized revenue growth score + normalized earnings growth score + normalized profit margin score + normalized ROE score + normalized debt to equity ratio score + normalized PEG ratio score) / 6
        # w1 = 20, w2 = 20, w3 = 10, w4 = 20, w5 = 15, w6 = 15
        vars = Config.ufs_needs + Config.uvs_needs + [Config.eveb, 'UVS', 'UFS']
        d = d[vars]
        d['10Score'] = 0
        my_thresholds = Config.ufs_thresholds
        my_thresholds.update(Config.uvs_thresholds)
        d = d.fillna(0)
        for m in my_thresholds.keys():
            d[m + '_norm'] = round((d[m] - d[m].min()) / (d[m].max() - d[m].min()), 2)
            d['10Score'] += d[m + '_norm']
        d['10Score'] = round(d['10Score']*10/11 , 2)
        d['Missing'] = 0
        return d.sort_values('10Score', ascending=False)

    # Value Score
    def uvs_scoring(self, d):
        # UVS = w1 * (normalized P / E) + w2 * (normalized P / B) + w3 * (normalized P / S) + w4 * (normalized EV / EBITDA) + w5 * (normalized Fwd P / E)
        # w1 = 20, w2 = 20, w3 = 10, w4 = 20, w5 = 30
        vars = Config.uvs_needs + Config.ufs_needs
        d = d[['symbol'] + vars]
        d.set_index('symbol', inplace=True)
        d[Config.eveb] = d[Config.ev] / d[Config.ebitda]
        my_thresholds = Config.uvs_thresholds
        for m in my_thresholds.keys():
            d[m + '_score'] = round((d[m].values[0] - my_thresholds[m][0]) / (my_thresholds[m][1] - my_thresholds[m][0]), 2)
            d[m + '_score'].fillna(0, inplace=True)
        d['UVS'] = round((Config.uvs_weights[Config.pe] * d[Config.pe + '_score'] + Config.uvs_weights[Config.pb] * d[Config.pb + '_score'] + Config.uvs_weights[Config.ps] * d[Config.ps + '_score'] +
                          Config.uvs_weights[Config.eveb] * d[Config.eveb + '_score'] + Config.uvs_weights[Config.fpe] * d[Config.fpe + '_score']) / 100, 2)
        return d.sort_values('UVS', ascending=False)


    def load_stocks_run_everything(self, tickers):
        self.tickers = []
        result = {}
        self.scores = {}
        try:
            for t in tickers:
                stock = yf.Ticker(t)
                self.ticker = stock
                self.tickers.append(stock)
                result[t] = {}
                company_info = self.get_company_info(t)
                financials = self.get_financial_metrics(t)
                result[t]['Company'] = company_info
                result[t]['Financials'] = financials
                start = '2015-01-01'
                plot_plain = self.plot_stock(t, start)
                plot_vs_sp500 = self.plot_stock_vs_sp500(t, start)
                result[t]['plot'] = plot_plain
                result[t]['plotvsp'] = plot_vs_sp500
            result['scores'] = self.score_stocks()
            return result
        except Exception as e:
            print(f"Error occurred: {e}")
            return {}


class MarketAnalysis():

    def get_main_plot(self, start):
        sp500_data = yf.download('^GSPC', start=start)
        nasdaq_data = yf.download('^IXIC', start=start) # ^BSESN
        normalized_sp500_data = sp500_data['Adj Close'] / sp500_data['Adj Close'].iloc[0]
        normalized_nasdaq_data = nasdaq_data['Adj Close'] / nasdaq_data['Adj Close'].iloc[0]

        plt.figure(figsize=(10, 6))
        plt.plot(normalized_sp500_data, label='SP500-Norm')
        plt.plot(normalized_nasdaq_data, label='NASDAQ-Norm')
        plt.xlabel('Date')
        plt.ylabel('Normalized Price')
        plt.legend()
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        return img_base64

    def run_market_analysis(self, start):
        result = {}
        img = self.get_main_plot(start)
        result['mainplot'] = img
        return result


class PlotOperator():

    def __init__(self):
        pass

