from pstats import Stats
import seaborn as sns
import panel as pn
pn.extension(loading_spinner='dots', loading_color='#00aa41', sizing_mode="stretch_width")
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
import plotly.figure_factory as ff

pn.extension("plotly", sizing_mode="stretch_width")

import matplotlib.font_manager

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import requests




PROJECT_ROOT = Path(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = str(PROJECT_ROOT.parent.parent)

#Import the golbal_feature_importance data from the folder model_and_data
globa_feature_importance = pd.read_csv(PROJECT_ROOT + '/model_and_data/golbal_feature_importance.csv', index_col=[0])
#"/Users/kone/Desktop/Oc_Formation/Projets/Projet7/openclassrooms_projects7/model_and_data/golbal_feature_importance.csv"
#"/Users/kone/Desktop/Oc_Formation/Projets/Projet7/openclassrooms_projects7/model_and_data/golbal_feature_importance.csv"


#Get the prediction probability from the API
#Get the prediction probability from the API
def get_pred_proba(SK_ID_CURR = 100001):
    """Get the prediction probability from the API"""

    url = 'http://127.0.0.1:1080/predict'  # localhost and the defined port + endpoint
    
    data = {'SK_ID_CURR': SK_ID_CURR}

    response = requests.post(url, data=data)
    
    return response.json()

#import test_df_not_norm data for the prediction
test_df_not_norm = pd.read_csv(PROJECT_ROOT + '/model_and_data/test_df_not_norm.csv', index_col=[0])
SK_ID_CURR_list = [str(x) for x in test_df_not_norm.sort_index().index.tolist()]


pn.extension(sizing_mode="stretch_width", template="fast")

def get_theme():
    return pn.state.session_args.get("theme", [b'default'])[0].decode()



theme=get_theme()

nice_accent_colors = [
    ("#00A170", "white"), # Mint
    ("#DAA520", "white"), # Golden Rod
    ("#F08080", "white"), # Light Coral
    ("#4099da", "white"), # Summery Sky
    ("#2F4F4F", "white"), # Dark Slate Grey
    ("#A01346", "white"), # Fast
]

def get_nice_accent_color():
    """Returns the 'next' nice accent color"""
    if not "color_index" in pn.state.cache:
        pn.state.cache["color_index"]=0
    elif pn.state.cache["color_index"]==len(nice_accent_colors)-1:
        pn.state.cache["color_index"]=0
    else:
        pn.state.cache["color_index"]+=1
    return nice_accent_colors[pn.state.cache["color_index"]]

accent_color, color = get_nice_accent_color()

def gauge_plot(trust_rate = 0):
    """Get the Trust value from the API and plot it as a gauge"""

    g = pn.indicators.Gauge(
    name='Acceptance rating', 
    value=trust_rate, 
    bounds=(0, 100), 
    format='{value} %',
    colors=[(0.6, 'red'), (0.8, 'gold'), (1, 'green')], 
    sizing_mode='stretch_width',
    margin=(0, 0, 0, 2),
    #height=200,
    #width=300,
    names=['low', 'medium', 'high'],
    )
    return g

#ploting the local feature importance
#ploting the local feature importance with plotly
def prediction_feature_importance_plot(
        SK_ID_CURR = '100001',
        nb_features = 15,
        template="simple_white",
        accent_color=accent_color,
        colorscale="Viridis",
        ):
    
    res = get_pred_proba(int(SK_ID_CURR))
    local_feature_importance = pd.DataFrame(res['local_explainer_df'])
    trust_rate = res['Trust rate']
    g = gauge_plot(trust_rate)

    df = local_feature_importance
    df['Importance'] = df['Importance'].round(3)
    
    #Bar plot with go
    local_feature_importance = go.Figure(

        data=[
            go.Bar(
                x=df[-nb_features:]['Importance'],
                y=df[-nb_features:]['Feature'],
                orientation="h",
                text=df[-nb_features:]['Importance'].astype(str),
                textposition='auto',
                textfont=dict(
                    family="sans serif",
                    size=14,
                ),
                marker=dict(
                    color=df[-nb_features:]['Importance'],
                    colorscale=colorscale,
                ),
                hovertemplate = "%{text}",
            )
        ],
        layout=go.Layout(
            #title=f"Local feature importance for client {SK_ID_CURR}",
            template=template,
            margin=dict(l=0, r=0, t=50, b=0),
            xaxis=dict(
                title="Local Importance",
                showgrid=False,
                showline=True,
                showticklabels=True,
                zeroline=True,
                linewidth=4,
                ticks='outside',
            ),
            yaxis=dict(
                title="Feature",
                showgrid=False,
                showline=True,
                showticklabels=True,
                zeroline=False,
                linewidth=2,
                ticks='outside',
            ),
        )
    )

    local_feature_importance.layout.autosize = True

    df = globa_feature_importance.sort_values('Importance', ascending=True)[-nb_features:]
    df['Importance'] = df['Importance'].round(3)
    
    #Bar plot with go
    global_feature_importance_plot = go.Figure(

        data=[
            go.Bar(
                x=df[-nb_features:]['Importance'],
                y=df[-nb_features:]['Feature'],
                orientation="h",
                text=df[-nb_features:]['Importance'].astype(str),
                textposition='auto',
                textfont=dict(
                    family="sans serif",
                    size=14,
                ),
                marker=dict(
                    color=df[-nb_features:]['Importance'],
                    colorscale=colorscale,
                ),
                hovertemplate = "%{text}",
            )
        ],
        layout=go.Layout(
            #title=f"Local feature importance for client {SK_ID_CURR}",
            template=template,
            margin=dict(l=0, r=0, t=50, b=0),
            xaxis=dict(
                title="Global Importance",
                showgrid=False,
                showline=True,
                showticklabels=True,
                zeroline=True,
                linewidth=4,
                ticks='outside',
            ),
            yaxis=dict(
                #title="Feature",
                showgrid=False,
                showline=True,
                showticklabels=True,
                zeroline=False,
                linewidth=2,
                ticks='outside',
            ),
        )
    )

    global_feature_importance_plot.layout.autosize = True

    #fig.layout.plot_bgcolor = accent_color 
    g_row = pn.Row(
        pn.pane.Markdown(
        """
        # The Gauge indicates the score obtained by the customer:
        - Red ==> Low Score (High default risk)
        - Gold ==> Average Score (Average Default Risk)
        - Green ==> High Score (Low risk of default)
        """,
        margin=(0, 0, 0, 0)
        ),
        g
    )
    local_feature_importance_column = pn.Column(
        pn.Row(
            pn.pane.Markdown(
                """
                ## The most influential Features for calculating the Customer Score:
                Features with a negative value decrease the probability that the customer is in default of payment while a positive value increases this probability.
                """, margin=(0, 0, 0, 0))
            ),
        pn.Row(local_feature_importance)
    )

    global_feature_importance_column = pn.Column(
        pn.Row(pn.pane.Markdown(
            """
            ## Global feature importance :
            ## The most important features for the model to predict the default risk of a customer.
            """, 
            margin=(0, 0, 0, 0))), 
        pn.Row(global_feature_importance_plot)
    )
    fig_column = pn.Row(local_feature_importance_column, global_feature_importance_column)
    return pn.Column(g_row, fig_column)


pn.Column(
    pn.Row(
        pn.layout.HSpacer(),
        pn.pane.PNG(
            "/Users/kone/Desktop/Oc_Formation/Projets/Projet7/openclassrooms_projects7/src/panel/logo.png",
            #sizing_mode="fixed",
            width=240            
        ),
        pn.layout.HSpacer(),
    ),
    pn.Row(pn.pane.Markdown('# Control Panel:'))
).servable(target="sidebar")


plotly_template = pn.widgets.Select(name="Select a Plotly Template", options=sorted(pio.templates))#.servable(target="sidebar")

if theme=="dark":
    plotly_template.value="plotly_dark"
else:
    plotly_template.value="simple_white"

colorscales = px.colors.named_colorscales()
colorscale = pn.widgets.Select(name="Select a color Palette", value="Viridis" ,options=colorscales)#.servable(target="sidebar")

slider_range = pn.widgets.EditableRangeSlider(name="Range", start=0, end=len(SK_ID_CURR_list), value=(0, 1000), step=1000, width=900)
SK_ID_CURR = pn.widgets.Select(value=str(SK_ID_CURR_list[0]), options=SK_ID_CURR_list[0:1000], width=120)

#update SK_ID_CURR when slider_range is changed
@pn.depends(slider_range.param.value, watch=True)
def update_SK_ID_CURR(value):
    SK_ID_CURR.options = SK_ID_CURR_list[value[0]:value[1]]
    SK_ID_CURR.value = str(SK_ID_CURR_list[value[0]])


pn.Row(
    pn.Column(pn.pane.Markdown('## Select a range of customers ID'), slider_range),# sizing_mode="stretch_width"),
    pn.Column(pn.pane.Markdown('## Select a customer ID'), SK_ID_CURR, sizing_mode="stretch_width"),
    sizing_mode="stretch_width"

).servable()

nb_features = pn.widgets.IntSlider(name="Number of Features", value=15, start=10, end=100, step=5)#.servable(target="sidebar")

#pn.Column(style, palette, font, nb_features, sizing_mode="stretch_width")#.servable(target="sidebar")

#vspacer = pn.layout.VSpacer(height=1)
#controler_title = pn.pane.Markdown('# Control Panel:').servable(target="sidebar")

pn.Column(
    pn.Row(plotly_template),
    pn.Row(colorscale),
    pn.Row(nb_features), 
    sizing_mode="stretch_width"
).servable(target="sidebar")

prediction_feature_importance_plot = pn.bind(
    prediction_feature_importance_plot, 
    SK_ID_CURR=SK_ID_CURR, 
    nb_features=nb_features,
    template=plotly_template,
    accent_color=accent_color,
    colorscale=colorscale,
    )

layout = pn.Column(prediction_feature_importance_plot, sizing_mode="stretch_width")

pn.panel(layout, sizing_mode="scale_width", loading=True).servable()

toggle_group = pn.widgets.ToggleGroup(
    name='Satistics', 
    options=['Feature Description', 'Categorical Features Distribution', 'Numerical Features Distribution'],  
    behavior="radio", 
    width=300,
    margin=(0, 0, 0, 0),
    sizing_mode="stretch_width",
    )

pn.Row(
    toggle_group
).servable()

#Import HomeCredit_columns_description.csv file
df_description = pd.read_csv(PROJECT_ROOT + "/data/tables/HomeCredit_columns_description.csv", encoding="ISO-8859-1", index_col=[0])

# Import cat_df.csv file
cat_df = pd.read_csv(PROJECT_ROOT + '/model_and_data/cat_df.csv', index_col=[0])

#Create a list of features
feature_names = list(df_description.Row)

#Create a list of categorical features
cat_feature_names = list(cat_df.columns)
cat_feature_names.remove('Dif_count')

#Import test_df_not_norm_not_norm.csv file and train_df_not_norm.csv file
test_df_not_norm = pd.read_csv(PROJECT_ROOT + '/model_and_data/test_df_not_norm.csv', index_col=[0])
train_df_not_norm = pd.read_csv(PROJECT_ROOT + '/model_and_data/train_df_not_norm.csv', index_col=[0])


#Create a list of numerical features
num_feature_names = []
for feat in test_df_not_norm.columns:

    if len(test_df_not_norm[feat].unique()) > 2:
        num_feature_names.append(feat)


def toggle_group_callback(toggle_group):


    if toggle_group == 'Feature Description':
        autocomplete = pn.widgets.AutocompleteInput(
        name='Feature Name', options=feature_names,
        placeholder='Enter a feature name',
        value = "TARGET",
        width=230,
        )

        def autocomplete_callback(autocomplete):

            try:
                row = df_description.loc[df_description['Row'] == autocomplete]
                description = row['Description'].values[0]
                table = row['Table'].values[0]
                special = row['Special'].values[0]

                card = pn.Card(
                    pn.pane.Markdown(f"""
                    ## {autocomplete}
                    ### Description: {description}
                    ### Table: {table}
                    ### Special: {special}
                    """, sizing_mode="stretch_width"),
                    #background=accent_color,
                    margin=(0, 0, 0, 0),
                    sizing_mode="stretch_width",
                    title="Feature Description",
                    #header_background='black',
                    #header_color='white',
                    border=True,
                    style={
                        'color': 'white',
                        'font-family': 'Helvetica',
                        'background-color': accent_color,
                        'border-color': accent_color,
                        'border-radius': '5px',
                    },
                )
                return card
            except:
                return pn.pane.Markdown(f"""
                ## {autocomplete}
                ### Description: Feature not found
                """, sizing_mode="stretch_width")
        
        return pn.Column(
            pn.Row(autocomplete),
            pn.Row(pn.bind(autocomplete_callback, autocomplete=autocomplete)),
            sizing_mode="stretch_width",
        )
    
    elif toggle_group == 'Categorical Features Distribution':
        """AutocompleteInput widget to select a categorical feature name and display the distribution of this feature"""

        autocomplete = pn.widgets.AutocompleteInput(
            name='Categorical Feature Name', options=cat_feature_names,
            placeholder='Enter a categorical feature name',
            value = 'CODE_GENDER',
            width=230,
        )

        def autocomplete_callback(autocomplete, color_continuous_scale='viridis'):

            df = cat_df[cat_df[autocomplete].notna()][[autocomplete, 'Dif_count']].sort_values(by='Dif_count', ascending=False)
            df.rename(columns={'Dif_count': 'Count(%)'}, inplace=True)

            fig = px.bar(
                df, 
                y=autocomplete, 
                x='Count(%)', 
                color='Count(%)', 
                color_continuous_scale=color_continuous_scale,
                orientation='h',
                height=400,
                width=850,
                )
            fig.update_layout(
                title=f"Distribution of '1 - client with payment difficulties' rate for {autocomplete} modalities",
                xaxis_title="Count(%)",
                yaxis_title=autocomplete,
                legend_title="Count(%)",
                )
            fig.update_xaxes(tickangle=45)
            #fig.update_yaxes(tickangle=45)
            fig.update_layout(legend=dict(
                orientation="h",
                yanchor='auto', #'auto', 'top', 'middle', 'bottom'
                x=1.02,
            ))
            fig.update_layout(
                margin=dict(l=0, r=0, t=30, b=0),
                paper_bgcolor=accent_color,
                plot_bgcolor=accent_color,
                font_color="white",

            )
            return fig
        
        return pn.Row(
            pn.Column(autocomplete),
            pn.Column(pn.bind(autocomplete_callback, autocomplete=autocomplete, color_continuous_scale=colorscale)),
            sizing_mode="stretch_width",
        )
    
    elif toggle_group == 'Numerical Features Distribution':
        """AutocompleteInput widget to select a numerical feature name and display the distribution of this feature and 
        mark the mean value, the std value and the median value.
         show the the client position in the distribution of this feature"""
        
        autocomplete = pn.widgets.AutocompleteInput(
            name='Numerical Feature Name', options=num_feature_names,
            placeholder='Enter a numerical feature name',
            value = num_feature_names[0],
            width=230,
        )

        nbins = pn.widgets.IntSlider(
            name='Number of bins', start=1, end=100, value=50,
            width=230,
        )

        log_y = pn.widgets.Checkbox(
            name='Log y-axis', value=False,
            width=230,
        )
        
        #Create a figure factory distribution plot for the selected feature
        def autocomplete_callback(autocomplete=num_feature_names[0],  SK_ID_CURR=100001, nbins=50, log_y=False):
            
        
            #Create a figure factory distribution plot for the selected feature
            try:
                
                fig = px.histogram(
                    train_df_not_norm, 
                    x=autocomplete, 
                    nbins=nbins, 
                    height=400, 
                    width=700,
                    marginal='violin',
                    color="TARGET",
                    log_y=log_y,
                    #histnorm='probability density',
                )

                fig.update_layout(
                    yaxis_title="Client Count"
                    )
                #Add go point for the client position in the histogram
                client_value = test_df_not_norm.loc[int(SK_ID_CURR), autocomplete]

                fig.add_trace(
                    go.Scatter(
                        x=[client_value],
                        y=[1],
                        mode="markers",
                        text=[SK_ID_CURR],
                        textposition="top center",
                        marker=dict(
                            color="black", 
                            size=12,
                            ),
                        name="Client value",
                        textfont=dict(
                            color="white",
                            size=12,
                        ),
                        texttemplate="%{text}"
                    )
                )
                
                fig.update_layout(
                    title=f"Distribution of {autocomplete}",
                    xaxis_title=autocomplete
                    )
                
                fig.update_layout(
                    margin=dict(l=0, r=0, t=30, b=0),
                    paper_bgcolor=accent_color,
                    plot_bgcolor=accent_color,
                    font_color="white",
                )

            
                
                #Add annotation for the client position in the histogram
                fig.add_annotation(
                    x=client_value,
                    y=0,
                    text=f"Client value: {client_value:.2f}",
                    showarrow=True,
                    arrowhead=1,
                    yshift=10,
                    )
                
                fig_pane = pn.pane.Plotly(fig, sizing_mode="stretch_width")
            
            except Exception as e:
                return pn.pane.Markdown(f"""
                ## {autocomplete}
                ### ERROR: {e}
                """, sizing_mode="stretch_width")
            
            return fig_pane
        
        fig_pane = pn.bind(autocomplete_callback, autocomplete=autocomplete, SK_ID_CURR=SK_ID_CURR, nbins=nbins, log_y=log_y)

        #Create a table with the client value for the selected feature
        # and the mean, std and median values

        def table_callback(autocomplete=autocomplete, SK_ID_CURR=SK_ID_CURR):

            df = train_df_not_norm[autocomplete].dropna()
            client_value = test_df_not_norm.loc[int(SK_ID_CURR), autocomplete]

            t_df = pd.DataFrame()
            t_df.loc['Client', 'Value'] = np.round(client_value, 2)
            t_df.loc['Mean', 'Value'] = np.round(df.mean(), 2)
            t_df.loc['Median', 'Value'] = np.round(df.median(), 2)
            t_df.loc['Std', 'Value'] = np.round(df.std(), 2)
            
            table = pn.pane.DataFrame(
                t_df,
                sizing_mode="stretch_width",
                style={
                    "border-color": accent_color,
                    "border-width": "4px",
                    "border-style": "solid",
                    "text-align": "center",
                    "font-size": "12pt",
                    "format": "{:.2f}",
                    "color": "white",
                    "background": "grey",
                }
            )
            
            #Add table in a card
            table = pn.Card(
                table,
                title="Statistics and client value",
                sizing_mode="stretch_width",
            )

            return table
        
        table = pn.bind(table_callback, autocomplete=autocomplete, SK_ID_CURR=SK_ID_CURR)

        
        return pn.Row(
            pn.Column(autocomplete, nbins, log_y),
            pn.Column(table),
            pn.Column(fig_pane),
            sizing_mode="stretch_width"
        )

    
pn.Column(
    pn.bind(toggle_group_callback, toggle_group=toggle_group),
    sizing_mode="stretch_width"
).servable()


pn.state.template.param.update(
    site="Prêt à dépenser", title="Dashboard", header_background=accent_color, accent_base_color=accent_color, favicon="https://raw.githubusercontent.com/mwaskom/seaborn/master/doc/_static/favicon.ico",
)

#panel serve --show --autoreload src/panel/dashoard.py --log-level=debug
