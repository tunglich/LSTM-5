# -*- coding: utf-8 -*-
import os
import io
import datetime
from dateutil.relativedelta import relativedelta
from finlab.data import Data
import sys
import numpy as np
#import matplotlib.pyplot as plt
import pickle
import time
#from sklearn.externals import joblib
import plotly.graph_objects as go
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.tools as tls
from dash.dependencies import Input, Output, State
#from scipy import stats
#import chart_studio.plotly as py
from plotly.subplots import make_subplots


group_colors = {"control": "light blue", "reference": "red"}

app = dash.Dash(

    __name__, meta_tags=[{"name": "viewport", "content": "width=device-width"}]

)

server = app.server

# 資料預處理 function
def ohlc_df(sid):
    close = data.get('收盤價', 100000)
    open_ = data.get('開盤價', 100000)
    high = data.get('最高價', 100000)
    low = data.get('最低價', 100000)
    volume = data.get('成交股數', 100000)

    # 判斷輸入為上市或是上櫃股票，找出其最新的 divide_ratio
    if sid in data.get('twse_divide_ratio').columns:
        adj_factor = data.get('twse_divide_ratio')[sid].dropna()[-1]
        adj_date = data.get('twse_divide_ratio')[sid].dropna().index[-1]
    else:
        adj_factor = data.get('otc_divide_ratio')[sid].dropna()[-1]
        adj_date = data.get('otc_divide_ratio')[sid].dropna().index[-1]

    # 將除權息日之前的股價資訊相除作為調整，除權息後股價資訊保持不變
    close_b = round(close[sid][:adj_date][:-1] / adj_factor, 1)
    close_a = close_b.append(close[sid][adj_date:])

    open_b = round(open_[sid][:adj_date][:-1] / adj_factor, 1)
    open_a = open_b.append(open_[sid][adj_date:])

    high_b = round(high[sid][:adj_date][:-1] / adj_factor, 1)
    high_a = high_b.append(high[sid][adj_date:])

    low_b = round(low[sid][:adj_date][:-1] / adj_factor, 1)
    low_a = low_b.append(low[sid][adj_date:])

    return pd.DataFrame({
        'stock_id': sid,
        'close': close_a,
        'open': open_a,
        'high': high_a,
        'low': low_a,
        'volume': volume[sid] / 1000

    })

def bar_color_v_function(index, price):
    real_date = datetime.datetime.strptime(index, '%d-%m-%Y').date()
    return "#006400" if price[real_date] < 0 else "#FF0000"
def bar_color_function(index, RF_p):
    real_date = datetime.datetime.strptime(index, '%d-%m-%Y').date()
    return "#FF0000" if RF_p[real_date] > 0.5 else "#006400"

# 求取有模型中所有股票 stock_id 含 ticker 及股票名稱
data = Data()

#pe = pd.read_pickle("history/tables/pe.pkl")
#filesRF = os.listdir("data/RF_pred")
#filesRF = [x[-8:-4] for x in filesRF]
#stock_id = [x for x in pe.index.get_level_values(0).unique().values if x[:4] in filesRF]

with open("history/stock_id.txt", "rb") as fp:
    stock_id = pickle.load(fp)

# App Layout
app.layout = html.Div(
    children=[
        # Error Message
        html.Div(id="error-message"),
        # Top Banner
        html.Div(
            className="study-browser-banner row",
            children=[
                html.H2(className="h2-title", children="TWSE AI MODEL - BETA 1.0"),
                html.Div(
                    className="div-logo",
                    children=html.Img(
                        className="logo", src=app.get_asset_url("dash-logo-new.png")
                    ),
                ),
                html.H2(className="h2-title-mobile", children="TWSE AI MODEL - BETA 1.0"),
            ],
        ),
        # Body of the App
        html.Div(
            className="row app-body",
            children=[
                # User Controls
                html.Div(
                    className="three columns card",
                    children=[
                        html.Div(
                            className="bg-white user-control",
                            children=[
                                html.Div(
                                    className="padding-top-bot",
                                    children=[
                                        html.H6("Stock Ticker"),
                                        html.Br(),
                                        dcc.Dropdown(placeholder='Enter/Select a Ticker', id="ticker-dropDown",
                                                     options=[{'label': i, 'value':i} for i in stock_id],
                                                     value='1101 台泥'

                                                     ),

                                    ], style={'fontSize': 12},
                                ),
                                html.Div(
                                    className="padding-top-bot",
                                    children=[
                                        html.H6("Type of Plot"),
                                        dcc.RadioItems(
                                            id="chart-type",
                                            options=[
                                                {"label": "LSTM Model", "value": "lstm"},
                                                {
                                                    "label": "Candle Stick",
                                                    "value": "candle",
                                                },
                                            ],
                                            value="candle",
                                            labelStyle={
                                                "display": "inline-block",
                                                "padding": "12px 12px 12px 0px",
                                            },
                                        ),
                                    ],
                                )
                            ],
                        )
                    ],
                ),
                # Graph
                html.Div(
                    className="nine columns card-left",
                    children=[
                        html.Div(
                            className="bg-white",
                            children=[
                                html.H5("Last Updated:"),
                                dcc.Input(id='h_date', value='0', type='hidden'),
                                html.Div(id='date', style={'marginLeft': '43px'}),
                                dcc.Graph(id="plot"),
                            ]
                        )
                    ],
                ),
                dcc.Store(id="error", storage_type="memory"),
            ],
        ),
    ]
)

df1 = ohlc_df('1101')
@app.callback(
    Output('date', 'children'),
    [Input('h_date', 'value')]
)
def update_date(value):
    date = df1.index[-1].date()
    return date

@app.callback(
    Output('plot', 'figure'),
    [Input('ticker-dropDown', 'value'),
     Input('chart-type', 'value')])
def update_graph(ticker, type):
    start_time = '2018-12-31'
    end_time = datetime.datetime.today()
    end_time = end_time.strftime("%Y-%m-%d")

    df = ohlc_df(ticker[:4])
    SMA60 = df['close'].rolling(60).mean()
    SMA20 = df['close'].rolling(20).mean()

    df = df[start_time:]
    SMA60 = SMA60[-len(df):]
    SMA20 = SMA20[-len(df):]

    # 將 RF 資料取出, 但由絕對路徑, 之後改為相對路徑
    AGGREGATION = pd.Series.from_csv('data/RF_pred/' + 'rf_pred_p_' + ticker[:4] + '.csv', parse_dates=True,
                                     index_col=0, header=None)
    AGGREGATION = AGGREGATION[start_time:]
    FUNDAMENTAL = pd.Series.from_csv('data/RF_pred/' + 'y_f_test_' + ticker[:4] + '.csv', parse_dates=True,
                                     index_col=0, header=None)
    FUNDAMENTAL = FUNDAMENTAL[start_time:]
    TECH_TREND = pd.Series.from_csv('data/RF_pred/' + 'y_tt_test_' + ticker[:4] + '.csv', parse_dates=True,
                                    index_col=0, header=None)
    TECH_TREND = TECH_TREND[start_time:]
    MOMENTUM = pd.Series.from_csv('data/RF_pred/' + 'y_m_test_' + ticker[:4] + '.csv', parse_dates=True,
                                  index_col=0, header=None)
    MOMENTUM = MOMENTUM[start_time:]
    TRADE = pd.Series.from_csv('data/RF_pred/' + 'y_t_test_' + ticker[:4] + '.csv', parse_dates=True,
                               index_col=0, header=None)
    TRADE = TRADE[start_time:]

    plot_date = df.index[-1].date().strftime("%Y-%m-%d")
    time_format = "%d-%m-%Y"

    # 為了處理假日資料會中斷的問題
    ticks = [date.strftime(time_format) for date in df.index]
    # 最多允許 250 筆資料
    space = max(int(len(ticks) / 100), 1)
    # 如果是無資料或超過範範的日期就去除
    for i, t in enumerate(ticks):
        ticks[i] = t if i % space == 0 or i == len(ticks) - 1 else ''

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.8, 0.2])
    trace1 = go.Candlestick(open=df['open'], high=df['high'],
                            low=df['low'], close=df['close'], x=ticks, increasing_line_color= 'red',
                            decreasing_line_color='green', name=ticker, showlegend=True)
    trace2 = go.Scatter(x=ticks, y=SMA20, mode='lines', name='SMA20', line=dict(color='orange', width=1),
                        showlegend=True)
    trace3 = go.Scatter(x=ticks, y=SMA60, mode='lines', name='SMA60', line=dict(color='royalblue', width=1),
                        showlegend=True)
    trace4 = go.Bar(x=ticks, y=df['volume'], name='Volume', showlegend=False)
    trace5 = go.Bar(x=ticks, y=AGGREGATION, name='AGGREGATION', showlegend=False)
    trace6 = go.Scatter(x=ticks, y=FUNDAMENTAL, mode='lines', name='FUNDAMENTAL',
                        line=dict(color='royalblue', width=1), showlegend=False)
    trace7 = go.Scatter(x=ticks, y=[0.5] * len(ticks), mode='lines',
                        hoverinfo='none', line=dict(color='black', dash='dot', width=1), showlegend=False)
    trace8 = go.Scatter(x=ticks, y=TECH_TREND, mode='lines', name='TECH_TREND',
                        line=dict(color='royalblue', width=1), showlegend=False)
    trace9 = go.Scatter(x=ticks, y=MOMENTUM, mode='lines', name='MOMENTUN',
                        showlegend=False, line=dict(color='royalblue', width=1))
    trace10 = go.Scatter(x=ticks, y=TRADE, mode='lines', name='TRADE',
                         showlegend=False, line=dict(color='royalblue', width=1))

    fig.add_trace(trace1, row=1, col=1)
    fig.add_trace(trace2, row=1, col=1)
    fig.add_trace(trace3, row=1, col=1)
    fig.add_trace(trace4, row=2, col=1)

    bar_color_v = [bar_color_v_function(j, df['close'].diff().fillna(0)) for j in ticks]

    fig.update_traces(marker_color=bar_color_v, row=2, col=1)
    fig.update_yaxes(tickfont=dict(size=9), row=2, col=1)
    fig.update_layout(legend_orientation='h', legend=dict(x=0.25, y=1.1))
    fig.update_layout(xaxis_rangeslider_visible=False, xaxis=dict(showgrid=False))
    fig.update_xaxes(ticks="inside", tickwidth=1, tickcolor='black', ticklen=5)
    fig.update_xaxes(nticks=25)
    fig.update_xaxes(tickfont=dict(size=9))
    fig.update_xaxes(showline=True, linewidth=0.5, linecolor='black')
    fig.update_yaxes(showline=True, linewidth=0.5, linecolor='black')
    fig.update_layout(height=600)

    fig.layout.template = 'plotly_white'

    if type == 'candle':
        return fig
    else:
        fig1 = make_subplots(rows=6, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                             row_heights=[0.40, 0.20, 0.10, 0.10, 0.10, 0.10])
        # price
        fig1.add_trace(trace1, row=1, col=1)
        fig1.add_trace(trace2, row=1, col=1)
        fig1.add_trace(trace3, row=1, col=1)
        # AGGREGATION
        fig1.add_trace(trace5, row=2, col=1)
        fig1.add_trace(trace7, row=2, col=1)
        # FUNDAMENTAL
        fig1.add_trace(trace6, row=3, col=1)
        fig1.add_trace(trace7, row=3, col=1)
        # TECH_TREND
        fig1.add_trace(trace8, row=4, col=1)
        fig1.add_trace(trace7, row=4, col=1)
        # MOMENTUM
        fig1.add_trace(trace9, row=5, col=1)
        fig1.add_trace(trace7, row=5, col=1)
        # TRADE
        fig1.add_trace(trace10, row=6, col=1)
        fig1.add_trace(trace7, row=6, col=1)

        bar_color = [bar_color_function(j, AGGREGATION) for j in ticks]

        fig1.update_traces(marker_color=bar_color, row=2, col=1)
        fig1.update_yaxes(tickfont=dict(size=9))
        fig1.update_yaxes(tickfont=dict(size=11), row=1, col=1)

        fig1.update_xaxes(ticks="inside", tickwidth=1, tickcolor='black', ticklen=5)
        fig1.update_xaxes(nticks=25)
        fig1.update_xaxes(tickfont=dict(size=9))
        fig1.update_xaxes(showline=True, linewidth=0.5, linecolor='black')
        fig1.update_yaxes(showline=True, linewidth=0.5, linecolor='black')
        fig1.update_layout(xaxis_rangeslider_visible=False, xaxis=dict(showgrid=False))
        fig1.update_layout(legend_orientation='h', legend=dict(x=0.25, y=1.1))
        fig1.update_layout(height=600)
        fig1.layout.template = 'plotly_white'

        return fig1

if __name__ == "__main__":
    app.run_server(debug=True)
