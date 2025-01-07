import json
from textwrap import dedent as d
from dash import Dash, dcc, html, Input, Output
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
from dash.dependencies import Input, Output


# app info
app = Dash(__name__,assets_folder ="static",
                 assets_url_path="static")

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}

# data and basic figure
y = np.arange(100)+10
x = pd.date_range(start='1/1/2021', periods=len(y))

fig = go.Figure(data=go.Scatter(x=x, y=y**2, mode = 'lines+markers'))
fig.add_traces(go.Scatter(x=x, y=y**2.2, mode = 'lines+markers'))

app.layout = html.Div([
    dcc.Graph(
        id='basic-interactions',
        figure=fig,
    ),

    html.Div(className='row', children=[
        html.Div([
            dcc.Markdown(d("""
              Click on points in the graph.
            """)),
            html.Pre(id='hover-data', style=styles['pre']),
        ], className='three columns'),
    ])
])


@app.callback(
    Output('hover-data', 'children'),
    [Input('basic-interactions', 'hoverData')])
def display_hover_data(hoverData):
    try:
        return json.dumps({'Date:':hoverData['points'][0],
                           'Value:':hoverData['points'][0]['y']}, indent = 2)
    except:
        return None

app.run_server(mode='external', port = 8080, dev_tools_ui=True,
          dev_tools_hot_reload =True, threaded=True)