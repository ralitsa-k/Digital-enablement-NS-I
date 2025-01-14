#%% Run setup
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import plotly.express as px
import dash_bootstrap_components as dbc
import os 
import dash
from textwrap import dedent as d 
from PIL import Image

CONTENT_STYLE = {
    "margin-left": "25rem",
    'backgorund-color': '#1e1e1e',
    "margin-top":'4rem'
}

app = Dash(__name__, external_stylesheets=[dbc.themes.SANDSTONE],assets_folder ="static",
                 assets_url_path="static")
server = app.server

header  = html.Div([ html.Div(
        className="app-header",
        children=[
            html.H2('Digital enablement user data',
                     className="app-header--title")])
        ]
    )

# Get the path for this current file 
curr_path = os.path.abspath(__file__)
# Get the root path by deleting everything after the specified folder 
curr_abs_path = curr_path.split('src')[0]
# Define paths for saving files and loading files 
save_path = curr_abs_path + '/data/'
digicare_df = pd.read_csv(save_path + '/LoginResponse.csv', sep=",", encoding='cp1252')

nsi_logo = Image.open(save_path+"NS&I_Sopra_Steria(1).png")

digicare_df['Date Case Closed'] = pd.to_datetime(digicare_df['Date Case Closed'], format="%d/%m/%Y")

first_day_of_data = min(digicare_df['Date Case Closed']).strftime("%A,%d %B %Y")

navbar = dbc.Navbar(
    dbc.Container(
        [
            html.A(
                dbc.Row(
                    [
                        dbc.Col(html.Img(src=nsi_logo, height="70px")),
                        dbc.Col([html.H3(["Digital Enablement - Data Stories"],className="ms-2" )]),
                    ]   
                )),
            dbc.NavbarToggler(id="navbar-toggler", n_clicks=0),
            dbc.Collapse(
                id="navbar-collapse",
                is_open=False,
                navbar=True,
            ),
        ]
    ),
    color='#d7d7d7',
    dark=True
)
 
colors = {"didn’t need Digicare":'#00585c',
            'only required a triage session':'#00585c',
            'required a data SIM':'#00585c',
            'required a device': '#00585c',
            'required a f2f coaching session': '#00585c',
            'required a remote coaching session': '#00585c'}

plot_figure = digicare_df.groupby(['Digicare Referrals'])\
    ['average log-ins to digital channels per month'].mean().reset_index()
fig = px.bar(plot_figure, x = 'average log-ins to digital channels per month',
             y = 'Digicare Referrals', hover_data='Digicare Referrals',
                         width=500, height=350)
fig.update_traces(hovertemplate='%{x}') #
fig.update_traces(marker_color='#00585c')
fig.update_layout(hovermode='closest',margin=dict(l=5, r=5, t=30, b=20), title='Digital Referrals')
fig.update_xaxes(title = 'Average count per month')
    
def get_month_from_datetime(df_and_colummn):
    return df_and_colummn.dt.to_period('M')

def drilldown_referred_to_channel(referral_type, digital_channel):
    # given the selected referral type, how many people logged in to digital channels
    # and within what timeframe
    login_df = digicare_df.loc[digicare_df['Digicare Referrals']==referral_type]
    login_df.loc[:,['Date logged in']] = pd.to_datetime(login_df['Date logged in'], format = "%d/%m/%Y")
    login_df.loc[:,['Days-passed']] = (pd.to_datetime(login_df['Date logged in']) - login_df['Date Case Closed']).dt.days
    login_df.loc[:,['Months-within']] = np.where((login_df.loc[:,['Days-passed']] > 270) & (login_df.loc[:,['Days-passed']] < 360), 'within 12 months',
                                                 np.where((login_df.loc[:,['Days-passed']] >180) & (login_df.loc[:,['Days-passed']] < 270), 'within 9 months',
                                                          np.where((login_df.loc[:,['Days-passed']] > 90) & (login_df.loc[:,['Days-passed']] < 180), 'within 6 months',
                                                                   np.where((login_df.loc[:,['Days-passed']] > 0) & (login_df.loc[:,['Days-passed']] < 90), 'within 3 months',np.nan))))
    impact_empty = pd.DataFrame()
    impact_empty['Digicare Impact'] = digicare_df['Digicare Impact'].unique()
    impact_empty['average log-ins to digital channels per month'] = 0
    plot_df = login_df.groupby('Digicare Impact')['average log-ins to digital channels per month'].mean().reset_index()
    df_diff = impact_empty[~impact_empty['Digicare Impact'].isin(plot_df['Digicare Impact'])]
    plot_df = pd.concat([plot_df, df_diff], ignore_index=True)
    fig_drilled = px.bar(plot_df, x = 'average log-ins to digital channels per month', y = 'Digicare Impact',
                         width=500, height=380)
    fig_drilled.update_traces(hovertemplate='%{x}') #
    fig_drilled.update_layout(margin=dict(t=30,l=5,r=5), title = 'Digital Impact')
    login_df_plotna = login_df.loc[login_df['Months-within']!= 'nan',:]
    login_df_plot = login_df_plotna.loc[login_df_plotna['Digicare Impact'] == digital_channel,:]
    
    months_empty = pd.DataFrame()
    months_empty['Months-within'] = login_df_plotna['Months-within'].unique()
    months_empty['average log-ins to digital channels per month'] = 0
    plot_month_count = login_df_plot.groupby(['Months-within'])['average log-ins to digital channels per month'].mean().reset_index()
    plot_month_count['average log-ins to digital channels per month'] = np.round(plot_month_count['average log-ins to digital channels per month'],2)
    df_diff2 = months_empty[~months_empty['Months-within'].isin(login_df_plot['Months-within'])]
    plot_month_count = pd.concat([plot_month_count, df_diff2], ignore_index=True)
    
    fig_month = px.bar(plot_month_count, x = 'Months-within', y = 'average log-ins to digital channels per month',
                       width=600, height=350, text_auto=True)
    fig_month.update_xaxes(title = 'Back to digital', categoryorder='array', categoryarray= ['within 3 months', 'within 6 months', 'within 9 months', 'within 12 months'])
    fig_month.update_yaxes(title = 'Count of returns')
    fig_month.update_traces(hovertemplate='%{x}', marker_color ='#fabd36') #
    return fig_drilled, fig_month
_, figbaselineMonth = drilldown_referred_to_channel('required a f2f coaching session', 'log in to digital channels')

def plot_registered():
    digicare_df['Date Case Opened'] = pd.to_datetime(digicare_df['Date Case Opened'], format = '%d/%m/%Y')
    digicare_df['Month-year']  = get_month_from_datetime(digicare_df['Date Case Opened'])
    plot_reg = digicare_df.loc[:,['Customer ID','Month-year']].drop_duplicates().groupby('Month-year')['Customer ID'].count().reset_index()
    plot_reg['Month-year'] = plot_reg['Month-year'].astype('str')
    fig = px.bar(plot_reg, x = 'Month-year', y = 'Customer ID')
    fig.update_traces(marker_color='#00c5ce')
    fig.update_yaxes(title='Count of customers')
    fig.update_layout(title='Digicare accounts opened')
    return fig

color_mapping = {'Online': '#c83a55',
                 'Social Media':'#ff496c',
                    'Webchat': '#ffba03',
                    'PolyAI': '#00c5ce',
                    'Poly AI': '#00c5ce',
                    'Phone': '#919191',
                    'Self-help': '#c1c1c6',
                    'Screenshare': '#05210a'}

def plot_channeld(color_mapping):
    contacts = pd.read_csv(save_path+'Contact.csv')
    contacts['Contact Date'] = pd.to_datetime(contacts['Contact Date'], format = "%d/%m/%Y")
    contacts['Contact Date Month'] = get_month_from_datetime(contacts['Contact Date'])
    cont_plot = contacts.drop_duplicates().groupby(['Contact Date Month','Contact Channel'])['Customer ID'].count().reset_index()
    cont_plot['Contact Date Month'] = cont_plot['Contact Date Month'].astype('str')
    
    fig = px.bar(cont_plot, x = 'Contact Date Month', y = 'Customer ID',
                 color = 'Contact Channel', barmode='group',color_discrete_map=color_mapping)
    fig.update_xaxes(tickangle= -45)
    fig.update_layout(title='Counts per Contact Channels')
    fig.update_layout(
        xaxis=dict(
            tickformat = '%b-%Y',
            tickvals=cont_plot['Contact Date Month'].unique()
        ))
    return fig

def plot_channel_containment(color_mapping):
    source = pd.read_csv(save_path+'Sankey(1).csv', encoding='cp1252')['Left axis'].tolist()
    target = pd.read_csv(save_path+'Sankey(1).csv',encoding='cp1252')['Right axis'].tolist()
    value = pd.read_csv(save_path+'Sankey(1).csv',encoding='cp1252')['Unnamed: 2'].tolist()
          
    all_namesRGB = pd.DataFrame(source + target)
    all_namesRGB['colors'] = all_namesRGB[0].map(color_mapping)
    # Function to convert a hex color to an RGB tuple
    # Create a Sankey diagram figure
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=10,
            thickness=20,
            line=dict(color="black", width=0.5),
            color="#b7b7b7",
            label=source + target, 
            hovertemplate='Count %{value}<extra></extra>'
        ),
        link=dict(
            source=[source.index(s) for s in source] + [len(source) + target.index(t) for t in target],
            target=[len(source) + target.index(t) for t in target] + [source.index(s) for s in source],
            value=value,
            hovercolor=all_namesRGB['colors'].to_list()
        )
    ),layout = go.Layout(title=dict(text="Channel Containment"), margin=dict(t=60)))
    return fig

def plot_personas_counts():
    df = pd.read_csv(save_path+'Customer.csv')
    df.groupby('Persona Nickname')['Customer ID'].count().reset_index()
#%% App layout
app.layout = html.Div([
    navbar,
    html.Div([
    html.Div(children = [html.Div([html.H3(f'Accounts Opened and Channel Interactions',
                                           className = 'header-boxes',
                                            style = {'margin':'2rem'})])]),
    html.Div(children = [html.Div([html.Div(f'Figures below show the number of accounts opened per month and the channels via which contact was initiated',
                     className = 'text-boxes', style = {'width':'40%','margin-left':'35px'})])]),
    html.Hr([]), 
    dbc.Row([dbc.Col([dcc.Graph(figure=plot_registered())]),
            dbc.Col([dcc.Graph(figure = plot_channel_containment(color_mapping))]),
            dbc.Col([dcc.Graph(figure = plot_channeld(color_mapping))])]),
    html.Hr([]),
    html.Div(children = [html.Div([html.H3(f'Digicare Referrals per month from {first_day_of_data} to date',
                                           className = 'header-boxes',style = {'margin':'2rem'})])]),
    html.Div(children = [html.Div([html.Div('Below are shown counts for digital referrals, digital impact for each (selected) digital referral, and counts of log-ins as a consequence of digital impact',
                     className = 'text-boxes', style = {'width':'40%','margin-left':'35px'})])]),
    html.Hr([]), 
    dbc.Row([dbc.Col([dcc.Graph(id='digital-referrals',figure=fig, hoverData={'points': [{'label': 'required a device'}]})]),
             dbc.Col([dcc.Graph(id='digital-referrals2')]),
             dbc.Col([html.Div(id='text1',style={'fontweight':'bold',
                                                          'font-size':'16px',
                                                          'margin-left':'20px',
                                                          'margin-right':'20px',
                                                          'width':'80%'}, className = 'text-boxes'),
                      dcc.Graph(id='month-plot', figure = figbaselineMonth)])]),
    html.Hr([])
    ], style={'margin':'60px'})
    ])


# Figure that changes based on hover
@app.callback(
    Output('digital-referrals2', 'figure'),
    Output('month-plot', 'figure'),
    Output('text1', 'children'),
    #Output('back-button', 'style'), #to hide/unhide the back button
    Input('digital-referrals', 'clickData'),
    Input('digital-referrals2', 'clickData'),    #for getting the vendor name from graph
    #Input('back-button', 'n_clicks')
)

def drilldown(clickFirst, clickSecond):

    if clickFirst is not None:
        referral = clickFirst['points'][0]['label']
        
        if referral in digicare_df['Digicare Referrals'].unique(): # if first plot is hovered over
            ctx = dash.callback_context
            trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
            if trigger_id == 'digital-referrals2': # if second plot was clicked
                if clickSecond is not None:
                    digital_channel = clickSecond['points'][0]['label']
                    fig1, figmonth  = drilldown_referred_to_channel(referral, digital_channel)
                    fig1.update_traces(marker_color='#00585c')
                    text1 = f'From those who {referral}, below are numbers of customers who {digital_channel}'
                    return fig1, figmonth, text1
                else: # if nothing clicked
                    fig1, figbaselineMonth = drilldown_referred_to_channel('required a f2f coaching session','log in to digital channels')
                    fig1.update_traces(marker_color='#00585c')
                    text1 = '[click on figure to filter]'
                    return fig1, figbaselineMonth, text1
            else: # if second plot was not clicked
                fig1, figbaselineMonth = drilldown_referred_to_channel(referral,'log in to digital channels')
                fig1.update_traces(marker_color='#00585c')
                text1 = '[click on figure to filter]'
                return fig1, figbaselineMonth, text1
    else:
        fig1, figbaselineMonth = drilldown_referred_to_channel('required a f2f coaching session','log in to digital channels')
        fig1.update_traces(marker_color='#00585c')
        text1 = '[click on figure to filter]'
        return fig1, figbaselineMonth, text1

if __name__ == '__main__':
    app.run_server(host= '0.0.0.0', port=8050)