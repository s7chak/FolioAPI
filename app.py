import base64
import io
import json
import logging
import os
import sys
from datetime import datetime as dt
from datetime import timedelta

import google.cloud.logging
import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
import plotly.io as pio
import yfinance as yf
from flask import Flask, request, jsonify, session, render_template_string, g
from sklearn.preprocessing import MinMaxScaler
from tsmoothie.smoother import DecomposeSmoother

from flask_session import Session
from ops.insights import ProcessOperator, NewsOperator, OptimalOperator

# client = google.cloud.logging.Client()
# client.setup_logging()

app = Flask(__name__)
app.config['SESSION_PERMANENT'] = True
app.config['SESSION_TYPE'] = 'filesystem'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=1)
Session(app)

envs = {
    'local' : {'type':'mac', 'url':''},
    'prod' : {'type':'gcp', 'url':'<CloudRun API Public URL>'}
}
active_env = 'local'
active_version = '4.0.1'

def delete_metadata():
    del session[g.code]['metadata']
    return


def load_fund_data(funds):
    for fund in funds:
        if 'metadata' in session[g.code] and fund in session[g.code]['metadata']:
            return
        else:
            get_fund_info(fund)


from google.cloud import storage
bucket_name = 'fols-buck'
def upload_blob(data, filename):
    global bucket_name
    path = f'gs://{bucket_name}/{filename}'
    try:
        data.to_csv(path)
    except:
        print("Failed to save file")

def read_file(stock_storage_file):
    global bucket_name
    if active_env!='local':
        client = storage.Client()
        blobs = client.list_blobs(bucket_name)
        try:
            for blob in blobs:
                if stock_storage_file == blob.name:
                    file_data = blob.download_as_bytes()
                    df_ = pd.read_csv(io.BytesIO(file_data))
                    return df_
                return None
        except Exception as e:
            print(str(e))
    else:
        return pd.read_csv(stock_storage_file)

def check_file_exists(stock_storage_file):
    global bucket_name
    if active_env != 'local':
        client = storage.Client()
        blobs = [b.name for b in client.list_blobs(bucket_name)]
        return stock_storage_file in blobs
    else:
        return os.path.isfile(stock_storage_file)


def process_stocks(stocks_str):
    df = pd.DataFrame(stocks_str)
    df = df[['name', 'quantity', 'cost', 'total']]
    df.drop('total', axis=1, inplace=True)
    today = str(dt.now().date())
    stock_storage_file = 'files/'+today+'.csv' if active_env=='local' else today+'.csv'
    if not check_file_exists(stock_storage_file):
        load_fund_data(list(df['name'].unique()))
        df['price'] = df['name'].apply(lambda x: session[g.code]['metadata'][x]['LastClosePrice'] if x in session[g.code]['metadata'] else np.nan)
        if active_env!='local':
            upload_blob(df[['name','price']], today + '.csv')
        else:
            df[['name', 'price']].to_csv('files/' + today + '.csv')
    else:
        read_df = read_file(stock_storage_file)
        if read_df.shape[0]==df.shape[0]:
            df = df.merge(read_df, on='name')[['name','quantity','cost','price']]
        else:
            load_fund_data(list(df['name'].unique()))
            df['price'] = df['name'].apply(lambda x: session[g.code]['metadata'][x]['LastClosePrice'] if x in session[g.code]['metadata'] else np.nan)
            if active_env != 'local':
                upload_blob(df[['name', 'price']], today + '.csv')
            else:
                df[['name', 'price']].to_csv('files/' + today + '.csv')
    # df['volatility'] = df['name'].apply(lambda x: round(session[g.code]['metadata'][x]['Volatility'],2) if session[g.code]['metadata'][x]['Volatility'] is not None else None)
    # df['peRatio'] = df['name'].apply(lambda x: session[g.code]['metadata'][x]['PE_Ratio'] if session[g.code]['metadata'][x]['PE_Ratio']!=np.nan else '')


    df['total_cost'] = round(df['cost']*df['quantity'],2)
    df['total_worth'] = round(df['price'] * df['quantity'],2)
    df['change'] = round(df['total_worth'] - df['total_cost'], 2)
    df['changepct'] = round((df['change'] * 100) / df['total_cost'], 2)
    total_portfolio_worth = df['total_worth'].sum()
    df['percentage'] = round((df['total_worth'] / total_portfolio_worth) * 100, 2)
    df.fillna(0, inplace=True)
    return df.to_dict('records')

def calculate_facts(stocks):
    df = pd.DataFrame(stocks)
    df['total_cost'] = df['cost'] * df['quantity']
    df['total_worth'] = df['price'] * df['quantity']
    total_cost = round(df['total_cost'].sum(),2)
    total_worth = round(df['total_worth'].sum(),2)
    result = {
        'Total Cost':  "${:.2f}".format(total_cost),
        'Total Worth': "${:.2f}".format(total_worth),
        'Change': "${:.2f}".format(total_worth - total_cost),
        'Change %': "{:.2f}".format(round((total_worth/total_cost - 1)*100,2))+"%"
    }
    return result

@app.route('/session_check', methods=['GET'])
def session_check():
    print("Session started...")
    print(session)
    res = {"message": "Session: "+str(session.keys())}
    return jsonify(res), 200


@app.route('/factsheet', methods=['POST'])
def factsheet():
    try:
        g.code = request.args.get('code')
        session[g.code] = {}
        stocks_data = request.json.get('stocks')
        stocks = process_stocks(stocks_data)
        session[g.code]['stocks'] = stocks
        # logging.info(str(session))
        result = calculate_facts(stocks)
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/stocksheet', methods=['GET'])
def stocksheet():
    try:
        g.code = request.args.get('code')
        gainSorted = request.args.get('gainSorted')
        stocks={}
        logging.info('Starting stock sheet fetch: '+str(session))
        if g.code in session:
            print('Session presence')
            stocks = session[g.code].get('stocks', [])
            sorting_options = {
                'change': lambda x: x['change'],
                'changepct': lambda x: x['changepct'],
                'percentage': lambda x: x['percentage'],
                'pe': lambda x: x['peRatio'],
                'volatility': lambda x: x['volatility']
            }
            if gainSorted in sorting_options:
                key_function = sorting_options[gainSorted]
                stocks = sorted(stocks, key=key_function, reverse=(gainSorted not in ['pe']))
        print('Post finding session.')
        return jsonify(stocks), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def load_historical(start):
    tickers = [s['name'] for s in session[g.code]['stocks']]
    start_date = dt.strptime(start, '%Y-%m-%dT%H:%M:%S.%fZ')
    combined_df = pd.DataFrame()
    tickers+=['^GSPC']
    path = 'files/'
    filename = 'storage.csv'
    data_present = False
    if os.path.isfile(path + filename):
        file_loaded = pd.read_csv(path + filename)
        if not file_loaded.empty:
            data_present = start.split('T')[0] >= file_loaded['Date'].min()
            for t in tickers:
                if t+'_Close' not in file_loaded.columns:
                    data_present = False
                    break
    if not data_present:
        for ticker in tickers:
            try:
                stock_data = yf.download(ticker, start=start_date - timedelta(days=1), end=pd.to_datetime('today'))
                stock_data = stock_data[['Close']]
                stock_data.columns = [
                    f"{ticker}_Close"
                ]
                if combined_df.empty:
                    combined_df = stock_data
                else:
                    combined_df = combined_df.join(stock_data)
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")
            combined_df.fillna(method='ffill').fillna(method='bfill')
            combined_df.to_csv('files/storage.csv')
    else:
        combined_df = file_loaded
        combined_df = combined_df.set_index('Date')
    combined_df = combined_df.fillna(method='ffill').fillna(method='bfill')
    session[g.code]['historical'] = combined_df

@app.route('/fetchHistory', methods=['GET'])
def fetchHistory():
    try:
        g.code = request.args.get('code')
        start_date = request.args.get('startDate')
        result = {}
        if g.code in session:
            history = session[g.code]['historical'] if 'historical' in session[g.code] else None
            if history is not None and not history.empty and type(history.index[0])==int:
                history = history.set_index('Date')
                session[g.code]['historical'] = history
            if history is None or (history.index.min() != start_date.split('T')[0]):
                load_historical(start_date)
                data = session[g.code]['historical']
                stocks = session[g.code]['stocks']
                for stock in stocks:
                    data[stock['name'] + '_Total'] = data[stock['name'] + '_Close'] * stock['quantity']
                data['Total'] = round(data[[stock['name'] + '_Total' for stock in stocks]].sum(axis=1), 2)
        result = {"message":"Loaded from "+start_date.split('T')[0]}
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e), "message": "Not loaded."}), 500


def correlation_plot(history):
    stock_names = [col.split('_')[0] for col in history.columns if col.endswith('_Close')]  + ['Total']
    closing_stock_names = [s + '_Close' for s in stock_names if s!='Total'] + ['Total']
    data = history[closing_stock_names]
    data.rename(columns={col: col.replace('_Close', '') for col in data.columns}, inplace=True)
    correlations = round(data.corr(),1)
    mask = np.triu(np.ones_like(correlations, dtype=bool))
    correlations = correlations.mask(mask)
    corr_values = correlations.values
    custom_text = np.where(~mask, corr_values.astype(str), '')
    # fig = px.imshow(correlations, x=stock_names, y=stock_names, labels=dict(color="Correlation"))
    fig = ff.create_annotated_heatmap(correlations.values, x=stock_names, y=stock_names, annotation_text=custom_text)
    start = history.index.min()
    fig.update_layout(
        title="Correlation Heatmap : since "+str(start),
        autosize=False,
        width=1400,
        height=1400,
    )
    img_bytes = pio.to_image(fig, format="png")
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    return img_base64


@app.route('/fetchCorrelation', methods=['GET'])
def fetchCorrelation():
    try:
        g.code = request.args.get('code')
        start = request.args.get('startDate')
        start_date = dt.strptime(start, '%Y-%m-%dT%H:%M:%S.%fZ').strftime('%Y-%m-%d')
        result = {"message": ""}
        if g.code in session and 'historical' in session[g.code]:
            history = session[g.code]['historical']
            data = history[start_date:]
            result['message'] = 'History loaded'
            img = correlation_plot(data)
            result['corr'] = img
        return jsonify(result), 200
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500


def anomaly_plot(data, stocks):
    stock_names = [col.split('_')[0] for col in data.columns if col.endswith('_Close')] + ['Total']
    closing_stock_names = [s + '_Close' for s in stock_names if s != 'Total'] + ['Total']
    data = data[closing_stock_names]
    data.rename(columns={col: col.replace('_Close', '') for col in data.columns}, inplace=True)
    stocks = stocks if stocks is not None else data.columns
    fig = go.Figure()
    smoother_dict = {}
    smooth_types = ['convolution', 'lowess', 'natural_cubic_spline']
    smooth_params = {
        'convolution': {'window_len':10, 'window_type': 'ones'},
        'lowess': {'smooth_fraction':0.1},
        'natural_cubic_spline': {}
    }
    smoother = DecomposeSmoother(periods=5, smooth_type=smooth_types[0], **smooth_params[smooth_types[0]])
    scaler = MinMaxScaler()
    for stock_symbol in stocks:
        stock_smoother = smoother.smooth(data[stock_symbol])
        smoothed_values = stock_smoother.smooth_data
        _low, _up = smoother.get_intervals('sigma_interval', n_sigma=2.5)
        stock_anomalies = list(np.where((data[stock_symbol].values < _low))[1]) + list(np.where((data[stock_symbol].values > _up))[1])
        scaled_series = scaler.fit_transform(data[stock_symbol].values.reshape(-1, 1))
        scaled_smoothed_values = smoother.smooth(scaled_series).smooth_data #scaler.inverse_transform(smoothed_values)
        _sc_low, _sc_up = smoother.get_intervals('sigma_interval', n_sigma=2.5)
        # stock_anomalies = list(np.where((scaled_series < _sc_low))[1]) + list(np.where((scaled_series > _sc_up))[1])

        # smooth_stock_anomalies = list(np.where((scaled_series < _sc_low))[1]) + list(np.where((scaled_series > _sc_up))[1])
        fig.add_trace(go.Scatter(x=data.index, y=pd.Series(scaled_series.T[0], index=data.index), mode='lines', name=f'{stock_symbol}', ))
        fig.add_trace(go.Scatter(x=data.index[stock_anomalies], y=pd.Series(scaled_smoothed_values[0][stock_anomalies]), mode='markers',
                                 name=f'{stock_symbol} Anomalies', marker=dict(color='red')))
        fig.add_trace(go.Scatter(x=data.index, y=pd.Series(_sc_low[0], index=data.index), mode='lines',
                                 name=f'{stock_symbol} Low', line=dict(color='green', width=1, dash='dash')))
        fig.add_trace(go.Scatter(x=data.index, y=pd.Series(_sc_up[0], index=data.index), mode='lines',
                                 name=f'{stock_symbol} High', line=dict(color='orange', width=1, dash='dash')))


        # Detect anomalies and store their indices
        # stock_anomalies = list(np.where((data[stock_symbol].values < _low))[1]) + list(np.where((data[stock_symbol].values > _up))[1])
        # fig.add_trace(go.Scatter(x=data.index, y=data[stock_symbol], mode='lines', name=f'{stock_symbol}',))
        # fig.add_trace(go.Scatter(x=data.index[stock_anomalies], y=smoothed_values[0][stock_anomalies], mode='markers',
        #                          name=f'{stock_symbol} Anomalies', marker=dict(color='red')))
        # fig.add_trace(go.Scatter(x=data.index, y=pd.Series(_low[0], index=data.index), mode='lines',
        #                          name=f'{stock_symbol} Low', line=dict(color='green', width=1, dash='dash')))
        # fig.add_trace(go.Scatter(x=data.index, y=pd.Series(_up[0], index=data.index), mode='lines',
        #                          name=f'{stock_symbol} High', line=dict(color='orange', width=1, dash='dash')))
    img_bytes = pio.to_image(fig, format="png")
    img_base64 = base64.b64encode(img_bytes).decode('utf-8')
    return img_base64

@app.route('/fetchAnomalies', methods=['GET'])
def fetchAnomalies():
    try:
        g.code = request.args.get('code')
        start = request.args.get('startDate')
        symbols_string = request.args.get('symbols')
        stock_symbols = symbols_string.split(',')
        start_date = dt.strptime(start, '%Y-%m-%dT%H:%M:%S.%fZ').strftime('%Y-%m-%d')
        result = {"message": ""}
        if stock_symbols==['']:
            result['message'] = 'Select some stocks.'
            return jsonify(result), 200
        if g.code in session and 'historical' in session[g.code]:
            history = session[g.code]['historical']
            data = history[start_date:]
            img = anomaly_plot(data, stock_symbols)
            result['anom'] = img
            result['message'] = 'Anomaly detection complete.'
        return jsonify(result), 200
    except Exception as e:
        print(e)
        return jsonify({"error": str(e), "message":"Not calculated."}), 500

def calculate_max_drawdown(historical_data, portfolio_stocks):
    daily_returns = historical_data.pct_change()
    peak_value = 0
    current_value = 0
    max_drawdown = 0

    for index, row in historical_data.iterrows():
        # Calculate the portfolio value for this day
        portfolio_value = 0
        for stock in portfolio_stocks:
            stock_name = stock['name']
            stock_quantity = stock['quantity']
            stock_price = row[f'{stock_name}_Close']
            portfolio_value += stock_quantity * stock_price

        # Update peak and current values
        if portfolio_value > peak_value:
            peak_value = portfolio_value
        else:
            drawdown = (peak_value - portfolio_value) / peak_value
            if drawdown > max_drawdown:
                max_drawdown = drawdown

    return max_drawdown

benchmark = '^GSPC'
def calculate_stock_metrics(history):
    portfolio_stocks = session[g.code]['stocks']
    weights = [stock['quantity'] for stock in portfolio_stocks]

    # Calculate daily returns
    historical_data = history
    benchmark_data = history[benchmark+'_Close']
    benchmark_returns = benchmark_data.pct_change()
    folio_returns = history['Total'].pct_change()

    if benchmark+'_Close' in history.columns:
        historical_data = history.drop(benchmark+'_Close', axis=1)
    if 'Total' in historical_data.columns:
        tot_cols = [s['name']+'_Total' for s in portfolio_stocks]
        total_worth = historical_data['Total']
        historical_data = historical_data.drop('Total', axis=1)
        historical_data = historical_data.drop(tot_cols, axis=1)

    found_stock = [h.split('_')[0] for h in historical_data]
    daily_returns = historical_data.pct_change()
    cumulative_returns = (1 + daily_returns).cumprod() - 1
    volatility = daily_returns.std()
    valid_mask = ~np.isnan(volatility)
    valid_volatility = volatility[valid_mask]
    valid_daily_returns = daily_returns[1:].loc[:, ~np.isnan(volatility)]
    valid_weights = [weights[i] for i in range(len(weights)) if valid_mask[i]]
    avg_daily_returns = valid_daily_returns.mean()
    start_price = historical_data.iloc[0][valid_mask]
    end_price = historical_data.iloc[-1][valid_mask]
    num_years = len(historical_data) / 252  # Assuming 252 trading days in a year
    cagr = (end_price / start_price) ** (1 / num_years) - 1
    risk_free_rate = 0.02
    sharpe_ratio = (cagr - risk_free_rate) / valid_volatility

    # Calculate the weighted average
    weighted_volatility = np.average(valid_volatility, weights=valid_weights, axis=0)
    weighted_sharpe_ratio = np.average(sharpe_ratio, weights=valid_weights)
    weighted_daily_ret = np.average(avg_daily_returns, weights=valid_weights)
    business_days_per_month = 21
    weighted_monthly_ret = round(weighted_daily_ret * business_days_per_month, 4)
    weighted_daily_ret = round(weighted_daily_ret, 4)
    weighted_cagr = round(np.average(cagr, weights=valid_weights), 2)
    folio_avg = {
        'Weighted Monthly Ret': weighted_monthly_ret,
        'Weighted Daily Ret': weighted_daily_ret,
        'Weighted CAGR': weighted_cagr,
        'Weighted Sharpe': round(weighted_sharpe_ratio, 2),
        'Weighted Volatility': round(weighted_volatility, 2),
    }
    ops = ProcessOperator(folio_returns, benchmark_returns)
    folio_stats = ops.calculate_stats()
    result = {'Avg':folio_avg, 'Stats':folio_stats}

    return result



@app.route('/fetchHistoryMetrics', methods=['GET'])
def fetchHistoryMetrics():
    try:
        g.code = request.args.get('code')
        start = request.args.get('startDate')
        start_date = dt.strptime(start, '%Y-%m-%dT%H:%M:%S.%fZ').strftime('%Y-%m-%d')
        result = {"message": ""}
        if g.code in session and 'historical' in session[g.code]:
            history = session[g.code]['historical']
            result['message'] = 'History analysed.'
            input_ = history[start_date:]
            insights = calculate_stock_metrics(input_)
            return jsonify(insights), 200
        return jsonify({"error": "No stocks"}), 500
    except Exception as e:
        print(e)
        return jsonify({"error": str(e)}), 500

@app.route('/showHistory', methods=['GET'])
def showHistory():
    try:
        g.code = request.args.get('code')
        start = request.args.get('startDate')
        start_date = dt.strptime(start, '%Y-%m-%dT%H:%M:%S.%fZ').strftime('%Y-%m-%d')
        res = []
        if g.code in session:
            history = session[g.code]['historical'] if 'historical' in session[g.code] else None
            stocks =  session[g.code]['stocks'] if 'stocks' in session[g.code] else None
            data = history[start_date:].reset_index()
            data.rename(columns={col: col.replace('_Close', '') for col in data.columns}, inplace=True)

            min_total = data['Total'].min()
            max_total = data['Total'].max()
            data['Folio'] = (data['Total'] - min_total) / (max_total - min_total)
            min_gspc = data['^GSPC'].min()
            max_gspc = data['^GSPC'].max()
            data['S&P'] = round((data['^GSPC'] - min_gspc) / (max_gspc - min_gspc),2)
            data = data.fillna(0)
            res = data.to_dict('records')
        result = {"history":res, "message":"Loaded from "+start_date.split('T')[0]}
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500



def get_fund_info(fund):
    try:
        ticker = yf.Ticker(fund.strip())
        fund_df = ticker.history(period='1d')
        if fund_df.empty:
            fund_df = ticker.history(period='2d')
        last_close_price = round(fund_df['Close'].iloc[-1], 2)
        # pe_ratio = ticker.info.get('trailingPE', None)
        # volatility = ticker.info.get('beta', None)
        metadata = {
            'LastClosePrice': last_close_price,
            # 'PE_Ratio': pe_ratio,
            # 'Volatility': volatility,
        }
        if 'metadata' not in session[g.code]:
            session[g.code]['metadata']={}
        session[g.code]['metadata'][fund] = metadata
        print('Fetched fund info:',fund, last_close_price)
        return metadata
    except Exception as e:
        print(f"Fund data not found: {fund} - {e}")
        return None




@app.route('/fetchOptimal', methods=['GET'])
def fetchOptimal():
    try:
        g.code = request.args.get('code')
        startDate = request.args.get('startDate')
        symbols_string = request.args.get('symbols')
        selected_stocks = symbols_string.split(',') if symbols_string.lower()!='all' else [s['name'] for s in session[g.code]['stocks']]
        result = {"message": ""}
        if g.code in session and 'historical' in session[g.code]:
            history = session[g.code]['historical']
            history.rename(columns={col: col.replace('_Close', '') for col in history.columns}, inplace=True)
            history = history[selected_stocks]
            returns = history.pct_change()
        else:
            result['message'] = "Please refetch historical data."
            return jsonify(result), 200
        op = OptimalOperator()
        metadata = session[g.code]['stocks']
        optimal_portfolio = op.calculate_optimal_frontier(metadata, returns, selected_stocks)
        result['optimal_folio'] = optimal_portfolio['optimal_weights']
        result['optimal'] = optimal_portfolio['efficient_frontier_plot']
        result['message'] = 'Optimal frontier calculation complete.\n'+'Start: '+startDate
        return jsonify(result), 200

    except Exception as e:
        print(e)
        return jsonify({"error": str(e), "message": "Failed to calculate optimal frontier."}), 500



@app.route('/headlines', methods=['GET'])
def fetch_headlines():
    try:
        g.code = request.args.get('code')
        session[g.code] = {}
        stocks_param = request.args.getlist('stocks[]')
        stocks = [json.loads(item) for item in stocks_param][0]
        result = {}
        if stocks!=[]:
            op = NewsOperator()
            try:
                result = op.headline_analysis(g.code, stocks)
            except:
                print(sys.exc_info())
        else:
            result['message'] = "Error getting stock symbols."
        return jsonify(result), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/', methods=['GET'])
def base():
    print("Session started")
    print(session)

    # Inline HTML template
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Folio API</title>
    </head>
    <body>
        <h1>Folio API</h1>
        <p>API is running.</p>
        <p>Version {active_version}</p>
    </body>
    </html>
    """

    return render_template_string(html_template)


@app.route('/health', methods=['GET'])
def health():
    success = "Server up and running.\n"+"Env: "+active_env +"\n"+ envs[active_env]['url']+"\nVersion: "+str(active_version)
    return jsonify({"message" : success})

app.secret_key = '<APPSECRET>'

if __name__ == '__main__':
    app.run(host="0.0.0.0", port="8091")
