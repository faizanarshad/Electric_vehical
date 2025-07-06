#!/usr/bin/env python3
"""
Enhanced Advanced Electric Vehicle Analysis Dashboard using Dash
Features: Modern UI, Machine Learning insights, Predictive Analytics, Advanced Visualizations
"""

import dash
from dash import dcc, html
from dash.dependencies import Output, Input
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data at the top
def load_data():
    df = pd.read_csv('electric_vehicles_spec_2025.csv.csv')
    numeric_columns = [
        'top_speed_kmh', 'battery_capacity_kWh', 'number_of_cells', 'torque_nm',
        'efficiency_wh_per_km', 'range_km', 'acceleration_0_100_s', 
        'fast_charging_power_kw_dc', 'towing_capacity_kg', 'cargo_volume_l',
        'seats', 'length_mm', 'width_mm', 'height_mm'
    ]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['segment'] = df['segment'].str.strip()
    df['segment_category'] = df['segment'].str.split(' - ').str[0]
    df['range_per_kwh'] = df['range_km'] / df['battery_capacity_kWh']
    df['power_to_weight_ratio'] = df['torque_nm'] / (df['length_mm'] * df['width_mm'] * df['height_mm'] * 1e-9)
    return df

df = load_data()

# Machine Learning Functions
def perform_clustering(df, n_clusters=4):
    features = ['range_km', 'top_speed_kmh', 'battery_capacity_kWh', 'efficiency_wh_per_km']
    data = df[features].dropna()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)
    data = data.copy()
    data['cluster'] = clusters
    return data

def predict_range(df, battery_capacity, efficiency):
    model_data = df[['battery_capacity_kWh', 'efficiency_wh_per_km', 'range_km']].dropna()
    X = model_data[['battery_capacity_kWh', 'efficiency_wh_per_km']]
    y = model_data['range_km']
    model = LinearRegression()
    model.fit(X, y)
    prediction = model.predict([[battery_capacity, efficiency]])
    return prediction[0], model.score(X, y)

# Initialize Dash app
app = dash.Dash(__name__, 
                title="🚗 Advanced EV Analytics Hub",
                suppress_callback_exceptions=True)

# Modern color scheme
colors = {
    'primary': '#1e3c72',
    'secondary': '#2a5298',
    'accent': '#667eea',
    'success': '#28a745',
    'warning': '#ffc107',
    'danger': '#dc3545',
    'light': '#f8f9fa',
    'dark': '#343a40',
    'gradient': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)'
}

# App layout with modern design
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("🚗 Advanced Electric Vehicle Analytics Hub", 
                style={'textAlign': 'center', 'color': 'white', 'marginBottom': '10px', 
                       'fontSize': '3rem', 'fontWeight': '700', 'textShadow': '2px 2px 4px rgba(0,0,0,0.3)'}),
        html.H3("Machine Learning Insights & Predictive Analytics", 
                style={'textAlign': 'center', 'color': 'rgba(255,255,255,0.9)', 'marginBottom': '20px',
                       'fontSize': '1.2rem', 'fontWeight': '400'})
    ], style={'background': colors['gradient'], 'padding': '2rem', 'borderRadius': '15px', 
              'marginBottom': '2rem', 'boxShadow': '0 8px 32px rgba(0,0,0,0.1)'}),
    
    # Main content
    dcc.Tabs([
        # Market Overview Tab
        dcc.Tab(label='📊 Market Overview', children=[
            html.Div([
                # Key metrics row
                html.Div([
                    html.Div([
                        html.H4("🚗 Total EVs", style={'textAlign': 'center', 'color': colors['primary'], 'fontSize': '1.2rem'}),
                        html.H2(id='total-evs', style={'textAlign': 'center', 'color': colors['primary'], 'fontSize': '2.5rem', 'fontWeight': '700'})
                    ], style={'background': 'rgba(255,255,255,0.95)', 'padding': '1.5rem', 'borderRadius': '15px', 
                              'margin': '10px', 'boxShadow': '0 4px 20px rgba(0,0,0,0.1)', 'border': '1px solid rgba(0,0,0,0.05)'}),
                    html.Div([
                        html.H4("🔋 Avg Range", style={'textAlign': 'center', 'color': colors['primary'], 'fontSize': '1.2rem'}),
                        html.H2(id='avg-range', style={'textAlign': 'center', 'color': colors['primary'], 'fontSize': '2.5rem', 'fontWeight': '700'})
                    ], style={'background': 'rgba(255,255,255,0.95)', 'padding': '1.5rem', 'borderRadius': '15px', 
                              'margin': '10px', 'boxShadow': '0 4px 20px rgba(0,0,0,0.1)', 'border': '1px solid rgba(0,0,0,0.05)'}),
                    html.Div([
                        html.H4("⚡ Avg Battery", style={'textAlign': 'center', 'color': colors['primary'], 'fontSize': '1.2rem'}),
                        html.H2(id='avg-battery', style={'textAlign': 'center', 'color': colors['primary'], 'fontSize': '2.5rem', 'fontWeight': '700'})
                    ], style={'background': 'rgba(255,255,255,0.95)', 'padding': '1.5rem', 'borderRadius': '15px', 
                              'margin': '10px', 'boxShadow': '0 4px 20px rgba(0,0,0,0.1)', 'border': '1px solid rgba(0,0,0,0.05)'}),
                    html.Div([
                        html.H4("🏁 Avg Speed", style={'textAlign': 'center', 'color': colors['primary'], 'fontSize': '1.2rem'}),
                        html.H2(id='avg-speed', style={'textAlign': 'center', 'color': colors['primary'], 'fontSize': '2.5rem', 'fontWeight': '700'})
                    ], style={'background': 'rgba(255,255,255,0.95)', 'padding': '1.5rem', 'borderRadius': '15px', 
                              'margin': '10px', 'boxShadow': '0 4px 20px rgba(0,0,0,0.1)', 'border': '1px solid rgba(0,0,0,0.05)'})
                ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '30px', 'flexWrap': 'wrap'}),
                
                # Charts row
                html.Div([
                    html.Div([
                        dcc.Graph(id='market-distribution', style={'height': '400px'})
                    ], style={'flex': '1', 'background': 'rgba(255,255,255,0.95)', 
                              'borderRadius': '15px', 'padding': '1rem', 'margin': '5px', 'boxShadow': '0 4px 20px rgba(0,0,0,0.1)'}),
                    html.Div([
                        dcc.Graph(id='performance-scatter', style={'height': '400px'})
                    ], style={'flex': '1', 'background': 'rgba(255,255,255,0.95)', 
                              'borderRadius': '15px', 'padding': '1rem', 'margin': '5px', 'boxShadow': '0 4px 20px rgba(0,0,0,0.1)'})
                ], style={'display': 'flex', 'justifyContent': 'space-between', 'gap': '10px'})
            ])
        ], style={'background': 'rgba(255,255,255,0.1)', 'borderRadius': '10px', 'margin': '5px'}),
        
        # ML Insights Tab
        dcc.Tab(label='🤖 ML Insights', children=[
            html.Div([
                html.Div([
                    html.Label("Number of Clusters:", style={'fontSize': '1.1rem', 'fontWeight': '600', 'color': colors['primary']}),
                    dcc.Slider(id='cluster-slider', min=2, max=6, value=4, marks={i: str(i) for i in range(2, 7)},
                              tooltip={"placement": "bottom", "always_visible": True})
                ], style={'width': '30%', 'margin': '20px', 'background': 'rgba(255,255,255,0.95)', 
                          'padding': '1rem', 'borderRadius': '10px', 'boxShadow': '0 4px 20px rgba(0,0,0,0.1)'}),
                html.Div([
                    html.Div([
                        dcc.Graph(id='cluster-scatter', style={'height': '400px'})
                    ], style={'flex': '1', 'background': 'rgba(255,255,255,0.95)', 
                              'borderRadius': '15px', 'padding': '1rem', 'margin': '5px', 'boxShadow': '0 4px 20px rgba(0,0,0,0.1)'}),
                    html.Div([
                        dcc.Graph(id='cluster-radar', style={'height': '400px'})
                    ], style={'flex': '1', 'background': 'rgba(255,255,255,0.95)', 
                              'borderRadius': '15px', 'padding': '1rem', 'margin': '5px', 'boxShadow': '0 4px 20px rgba(0,0,0,0.1)'})
                ], style={'display': 'flex', 'justifyContent': 'space-between', 'gap': '10px'}),
                html.Div(id='cluster-insights', style={'marginTop': '20px', 'background': 'rgba(255,255,255,0.95)', 
                                                       'padding': '1rem', 'borderRadius': '10px', 'boxShadow': '0 4px 20px rgba(0,0,0,0.1)'})
            ])
        ], style={'background': 'rgba(255,255,255,0.1)', 'borderRadius': '10px', 'margin': '5px'}),
        
        # Predictive Analytics Tab
        dcc.Tab(label='🔮 Predictive Analytics', children=[
            html.Div([
                html.Div([
                    html.Div([
                        html.Label("Battery Capacity (kWh):", style={'fontSize': '1.1rem', 'fontWeight': '600', 'color': colors['primary']}),
                        dcc.Slider(id='battery-slider', min=30, max=150, value=75, step=5,
                                  tooltip={"placement": "bottom", "always_visible": True})
                    ], style={'width': '30%', 'float': 'left', 'background': 'rgba(255,255,255,0.95)', 
                              'padding': '1rem', 'borderRadius': '10px', 'margin': '5px', 'boxShadow': '0 4px 20px rgba(0,0,0,0.1)'}),
                    html.Div([
                        html.Label("Efficiency (Wh/km):", style={'fontSize': '1.1rem', 'fontWeight': '600', 'color': colors['primary']}),
                        dcc.Slider(id='efficiency-slider', min=100, max=300, value=200, step=10,
                                  tooltip={"placement": "bottom", "always_visible": True})
                    ], style={'width': '30%', 'float': 'left', 'background': 'rgba(255,255,255,0.95)', 
                              'padding': '1rem', 'borderRadius': '10px', 'margin': '5px', 'boxShadow': '0 4px 20px rgba(0,0,0,0.1)'}),
                    html.Div([
                        html.Label("Predicted Range:", style={'fontSize': '1.1rem', 'fontWeight': '600', 'color': colors['primary']}),
                        html.H3(id='predicted-range', style={'color': colors['success'], 'fontSize': '2rem', 'fontWeight': '700'})
                    ], style={'width': '30%', 'float': 'left', 'background': 'rgba(255,255,255,0.95)', 
                              'padding': '1rem', 'borderRadius': '10px', 'margin': '5px', 'boxShadow': '0 4px 20px rgba(0,0,0,0.1)'})
                ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '30px', 'clear': 'both'}),
                html.Div([
                    html.Div([
                        dcc.Graph(id='prediction-scatter', style={'height': '400px'})
                    ], style={'flex': '1', 'background': 'rgba(255,255,255,0.95)', 
                              'borderRadius': '15px', 'padding': '1rem', 'margin': '5px', 'boxShadow': '0 4px 20px rgba(0,0,0,0.1)'}),
                    html.Div([
                        dcc.Graph(id='prediction-surface', style={'height': '400px'})
                    ], style={'flex': '1', 'background': 'rgba(255,255,255,0.95)', 
                              'borderRadius': '15px', 'padding': '1rem', 'margin': '5px', 'boxShadow': '0 4px 20px rgba(0,0,0,0.1)'})
                ], style={'display': 'flex', 'justifyContent': 'space-between', 'gap': '10px'})
            ])
        ], style={'background': 'rgba(255,255,255,0.1)', 'borderRadius': '10px', 'margin': '5px'}),
        
        # Advanced Analytics Tab
        dcc.Tab(label='📈 Advanced Analytics', children=[
            html.Div([
                html.Div([
                    html.H3("Feature Correlation Analysis", style={'color': colors['primary'], 'textAlign': 'center'}),
                    dcc.Graph(id='correlation-heatmap', style={'height': '500px'})
                ], style={'background': 'rgba(255,255,255,0.95)', 'borderRadius': '15px', 'padding': '1rem', 
                          'margin': '10px', 'boxShadow': '0 4px 20px rgba(0,0,0,0.1)'}),
                html.Div([
                    html.Div([
                        dcc.Graph(id='distribution-plots', style={'height': '500px'})
                    ], style={'flex': '1', 'background': 'rgba(255,255,255,0.95)', 
                              'borderRadius': '15px', 'padding': '1rem', 'margin': '5px', 'boxShadow': '0 4px 20px rgba(0,0,0,0.1)'}),
                    html.Div([
                        dcc.Graph(id='box-plots', style={'height': '500px'})
                    ], style={'flex': '1', 'background': 'rgba(255,255,255,0.95)', 
                              'borderRadius': '15px', 'padding': '1rem', 'margin': '5px', 'boxShadow': '0 4px 20px rgba(0,0,0,0.1)'})
                ], style={'display': 'flex', 'justifyContent': 'space-between', 'gap': '10px'})
            ])
        ], style={'background': 'rgba(255,255,255,0.1)', 'borderRadius': '10px', 'margin': '5px'}),
        
        # Interactive Explorer Tab
        dcc.Tab(label='🔍 Interactive Explorer', children=[
            html.Div([
                html.Div([
                    html.Div([
                        html.Label("Brand:", style={'fontSize': '1.1rem', 'fontWeight': '600', 'color': colors['primary']}),
                        dcc.Dropdown(id='brand-dropdown', multi=True, 
                                   options=[{'label': b, 'value': b} for b in sorted(df['brand'].unique())],
                                   value=sorted(df['brand'].unique())[:5])
                    ], style={'width': '25%', 'float': 'left', 'background': 'rgba(255,255,255,0.95)', 
                              'padding': '1rem', 'borderRadius': '10px', 'margin': '5px', 'boxShadow': '0 4px 20px rgba(0,0,0,0.1)'}),
                    html.Div([
                        html.Label("Segment:", style={'fontSize': '1.1rem', 'fontWeight': '600', 'color': colors['primary']}),
                        dcc.Dropdown(id='segment-dropdown', multi=True, 
                                   options=[{'label': s, 'value': s} for s in sorted(df['segment_category'].unique())],
                                   value=sorted(df['segment_category'].unique()))
                    ], style={'width': '25%', 'float': 'left', 'background': 'rgba(255,255,255,0.95)', 
                              'padding': '1rem', 'borderRadius': '10px', 'margin': '5px', 'boxShadow': '0 4px 20px rgba(0,0,0,0.1)'}),
                    html.Div([
                        html.Label("Min Range (km):", style={'fontSize': '1.1rem', 'fontWeight': '600', 'color': colors['primary']}),
                        dcc.Slider(id='range-slider', min=0, max=700, value=0,
                                  tooltip={"placement": "bottom", "always_visible": True})
                    ], style={'width': '25%', 'float': 'left', 'background': 'rgba(255,255,255,0.95)', 
                              'padding': '1rem', 'borderRadius': '10px', 'margin': '5px', 'boxShadow': '0 4px 20px rgba(0,0,0,0.1)'}),
                    html.Div([
                        html.Label("Min Speed (km/h):", style={'fontSize': '1.1rem', 'fontWeight': '600', 'color': colors['primary']}),
                        dcc.Slider(id='speed-slider', min=0, max=350, value=0,
                                  tooltip={"placement": "bottom", "always_visible": True})
                    ], style={'width': '25%', 'float': 'left', 'background': 'rgba(255,255,255,0.95)', 
                              'padding': '1rem', 'borderRadius': '10px', 'margin': '5px', 'boxShadow': '0 4px 20px rgba(0,0,0,0.1)'})
                ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '20px', 'clear': 'both'}),
                html.Div([
                    html.Div([
                        dcc.Graph(id='interactive-scatter', style={'height': '400px'})
                    ], style={'flex': '1', 'background': 'rgba(255,255,255,0.95)', 
                              'borderRadius': '15px', 'padding': '1rem', 'margin': '5px', 'boxShadow': '0 4px 20px rgba(0,0,0,0.1)'}),
                    html.Div([
                        dcc.Graph(id='interactive-3d', style={'height': '400px'})
                    ], style={'flex': '1', 'background': 'rgba(255,255,255,0.95)', 
                              'borderRadius': '15px', 'padding': '1rem', 'margin': '5px', 'boxShadow': '0 4px 20px rgba(0,0,0,0.1)'})
                ], style={'display': 'flex', 'justifyContent': 'space-between', 'gap': '10px'}),
                html.Div([
                    html.H3("Filtered Data", style={'color': colors['primary'], 'textAlign': 'center'}),
                    html.Div(id='data-table', style={'background': 'rgba(255,255,255,0.95)', 'borderRadius': '10px', 
                                                     'padding': '1rem', 'boxShadow': '0 4px 20px rgba(0,0,0,0.1)'})
                ])
            ])
        ], style={'background': 'rgba(255,255,255,0.1)', 'borderRadius': '10px', 'margin': '5px'})
    ], style={'background': 'rgba(255,255,255,0.1)', 'borderRadius': '10px', 'margin': '5px'})
], style={'background': colors['gradient'], 'minHeight': '100vh', 'padding': '2rem'})

# Callbacks with enhanced styling
@app.callback(
    [Output('total-evs', 'children'),
     Output('avg-range', 'children'),
     Output('avg-battery', 'children'),
     Output('avg-speed', 'children')],
    [Input('market-distribution', 'id')]
)
def update_metrics(_):
    return (
        f"{len(df):,}",
        f"{df['range_km'].mean():.0f} km",
        f"{df['battery_capacity_kWh'].mean():.1f} kWh",
        f"{df['top_speed_kmh'].mean():.0f} km/h"
    )

@app.callback(
    Output('market-distribution', 'figure'),
    [Input('performance-scatter', 'id')]
)
def update_market_distribution(_):
    brand_counts = df['brand'].value_counts().head(10)
    fig = px.pie(
        values=brand_counts.values,
        names=brand_counts.index,
        title="🚗 Top 10 Brands Distribution",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        title_font_size=20,
        title_font_color=colors['primary'],
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

@app.callback(
    Output('performance-scatter', 'figure'),
    [Input('market-distribution', 'id')]
)
def update_performance_scatter(_):
    fig = px.scatter(
        df, x='range_km', y='top_speed_kmh',
        color='brand', size='battery_capacity_kWh',
        hover_data=['model', 'segment_category'],
        title="⚡ Performance Analysis: Range vs Speed",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig.update_layout(
        xaxis_title="Range (km)",
        yaxis_title="Top Speed (km/h)",
        title_font_size=20,
        title_font_color=colors['primary'],
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

@app.callback(
    [Output('cluster-scatter', 'figure'),
     Output('cluster-radar', 'figure'),
     Output('cluster-insights', 'children')],
    [Input('cluster-slider', 'value')]
)
def update_clustering(n_clusters):
    clustered_data = perform_clustering(df, n_clusters)
    
    # Scatter plot
    fig_scatter = px.scatter(
        clustered_data, x='range_km', y='top_speed_kmh',
        color='cluster', size='battery_capacity_kWh',
        title=f"🤖 EV Clusters (K={n_clusters})",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig_scatter.update_layout(
        title_font_size=20,
        title_font_color=colors['primary'],
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    # Radar chart
    cluster_centers = clustered_data.groupby('cluster').mean()
    fig_radar = go.Figure()
    for cluster in range(n_clusters):
        center = cluster_centers.loc[cluster]
        fig_radar.add_trace(go.Scatterpolar(
            r=[center['range_km'], center['top_speed_kmh'], 
               center['battery_capacity_kWh'], center['efficiency_wh_per_km']],
            theta=['Range', 'Speed', 'Battery', 'Efficiency'],
            fill='toself',
            name=f'Cluster {cluster}',
            line_color=px.colors.qualitative.Set3[cluster]
        ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, cluster_centers.max().max()])),
        title="📊 Cluster Characteristics",
        title_font_size=20,
        title_font_color=colors['primary'],
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    # Insights
    insights = []
    for cluster in range(n_clusters):
        cluster_data = clustered_data[clustered_data['cluster'] == cluster]
        insights.append(html.Div([
            html.H4(f"Cluster {cluster} ({len(cluster_data)} vehicles)", 
                   style={'color': colors['primary'], 'marginBottom': '10px'}),
            html.P(f"🔋 Avg Range: {cluster_data['range_km'].mean():.0f} km"),
            html.P(f"🏁 Avg Speed: {cluster_data['top_speed_kmh'].mean():.0f} km/h"),
            html.P(f"⚡ Avg Battery: {cluster_data['battery_capacity_kWh'].mean():.1f} kWh")
        ], style={'background': 'rgba(255,255,255,0.8)', 'padding': '1rem', 'borderRadius': '10px', 
                  'margin': '10px', 'borderLeft': f'4px solid {px.colors.qualitative.Set3[cluster]}'}))
    
    return fig_scatter, fig_radar, insights

@app.callback(
    [Output('predicted-range', 'children'),
     Output('prediction-scatter', 'figure'),
     Output('prediction-surface', 'figure')],
    [Input('battery-slider', 'value'),
     Input('efficiency-slider', 'value')]
)
def update_predictions(battery, efficiency):
    predicted_range, r2_score_val = predict_range(df, battery, efficiency)
    
    # Scatter plot
    model_data = df[['battery_capacity_kWh', 'efficiency_wh_per_km', 'range_km']].dropna()
    fig_scatter = px.scatter(
        model_data, x='battery_capacity_kWh', y='range_km',
        color='efficiency_wh_per_km',
        title=f"🔮 Range Prediction Model (R² = {r2_score_val:.3f})",
        color_continuous_scale='viridis'
    )
    fig_scatter.add_trace(go.Scatter(
        x=[battery], y=[predicted_range],
        mode='markers',
        marker=dict(size=20, color='red', symbol='star'),
        name='Prediction'
    ))
    fig_scatter.update_layout(
        title_font_size=20,
        title_font_color=colors['primary'],
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    # Surface plot
    x_range = np.linspace(30, 150, 20)
    y_range = np.linspace(100, 300, 20)
    X, Y = np.meshgrid(x_range, y_range)
    model = LinearRegression()
    X_train = model_data[['battery_capacity_kWh', 'efficiency_wh_per_km']]
    y_train = model_data['range_km']
    model.fit(X_train, y_train)
    Z = model.predict(np.column_stack([X.ravel(), Y.ravel()])).reshape(X.shape)
    
    fig_surface = go.Figure(data=[go.Surface(x=X, y=Y, z=Z, colorscale='viridis')])
    fig_surface.update_layout(
        title="🌐 Range Prediction Surface",
        scene=dict(
            xaxis_title="Battery Capacity (kWh)",
            yaxis_title="Efficiency (Wh/km)",
            zaxis_title="Predicted Range (km)"
        ),
        title_font_size=20,
        title_font_color=colors['primary']
    )
    
    return f"{predicted_range:.0f} km", fig_scatter, fig_surface

@app.callback(
    Output('correlation-heatmap', 'figure'),
    [Input('correlation-heatmap', 'id')]
)
def update_correlation_heatmap(_):
    features = ['range_km', 'top_speed_kmh', 'battery_capacity_kWh', 
                'efficiency_wh_per_km', 'acceleration_0_100_s']
    corr_matrix = df[features].corr()
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="📊 Feature Correlation Matrix",
        color_continuous_scale="RdBu"
    )
    fig.update_layout(
        title_font_size=20,
        title_font_color=colors['primary'],
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

@app.callback(
    [Output('distribution-plots', 'figure'),
     Output('box-plots', 'figure')],
    [Input('distribution-plots', 'id')]
)
def update_distribution_plots(_):
    # Distribution plots
    fig_dist = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Range Distribution', 'Speed Distribution', 
                       'Battery Distribution', 'Efficiency Distribution')
    )
    fig_dist.add_trace(go.Histogram(x=df['range_km'].dropna(), name='Range'), row=1, col=1)
    fig_dist.add_trace(go.Histogram(x=df['top_speed_kmh'].dropna(), name='Speed'), row=1, col=2)
    fig_dist.add_trace(go.Histogram(x=df['battery_capacity_kWh'].dropna(), name='Battery'), row=2, col=1)
    fig_dist.add_trace(go.Histogram(x=df['efficiency_wh_per_km'].dropna(), name='Efficiency'), row=2, col=2)
    fig_dist.update_layout(
        height=600, 
        showlegend=False,
        title_font_size=20,
        title_font_color=colors['primary'],
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    # Box plots
    fig_box = px.box(
        df, x='segment_category', y='range_km',
        title="📦 Range Distribution by Segment",
        color_discrete_sequence=[colors['primary']]
    )
    fig_box.update_layout(
        title_font_size=20,
        title_font_color=colors['primary'],
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig_dist, fig_box

@app.callback(
    [Output('interactive-scatter', 'figure'),
     Output('interactive-3d', 'figure'),
     Output('data-table', 'children')],
    [Input('brand-dropdown', 'value'),
     Input('segment-dropdown', 'value'),
     Input('range-slider', 'value'),
     Input('speed-slider', 'value')]
)
def update_interactive_explorer(brands, segments, min_range, min_speed):
    filtered_df = df.copy()
    if brands:
        filtered_df = filtered_df[filtered_df['brand'].isin(brands)]
    if segments:
        filtered_df = filtered_df[filtered_df['segment_category'].isin(segments)]
    filtered_df = filtered_df[
        (filtered_df['range_km'] >= min_range) &
        (filtered_df['top_speed_kmh'] >= min_speed)
    ]
    
    # 2D scatter
    fig_scatter = px.scatter(
        filtered_df, x='top_speed_kmh', y='range_km',
        color='brand', size='battery_capacity_kWh',
        hover_data=['model', 'segment_category'],
        title=f"🔍 Filtered Data ({len(filtered_df)} vehicles)",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig_scatter.update_layout(
        title_font_size=20,
        title_font_color=colors['primary'],
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    # 3D scatter
    fig_3d = px.scatter_3d(
        filtered_df, x='battery_capacity_kWh', y='range_km', z='top_speed_kmh',
        color='segment_category', size='efficiency_wh_per_km',
        hover_data=['brand', 'model'],
        title="🌐 3D View: Battery vs Range vs Speed"
    )
    fig_3d.update_layout(
        title_font_size=20,
        title_font_color=colors['primary']
    )
    
    # Data table
    table_data = filtered_df[['brand', 'model', 'range_km', 'top_speed_kmh', 'battery_capacity_kWh']].head(20)
    table = html.Table([
        html.Thead(html.Tr([html.Th(col, style={'color': colors['primary'], 'fontWeight': '600'}) for col in table_data.columns])),
        html.Tbody([
            html.Tr([html.Td(table_data.iloc[i][col]) for col in table_data.columns])
            for i in range(len(table_data))
        ])
    ], style={'width': '100%', 'borderCollapse': 'collapse'})
    
    return fig_scatter, fig_3d, table

# Custom CSS for enhanced styling
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        <title>🚗 Advanced EV Analytics Hub</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 0;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }
            .dash-tab {
                background: rgba(255,255,255,0.8);
                border-radius: 10px 10px 0 0;
                border: none;
                padding: 1rem 2rem;
                font-weight: 600;
                transition: all 0.3s ease;
                margin: 0 5px;
            }
            .dash-tab--selected {
                background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
                color: white;
                box-shadow: 0 4px 15px rgba(30,60,114,0.3);
            }
            .dash-tab:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            }
            .dash-tab-list {
                background: rgba(255,255,255,0.1);
                border-radius: 10px;
                margin: 10px;
                padding: 5px;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

if __name__ == '__main__':
    app.run(debug=True, host='localhost', port=8050) 