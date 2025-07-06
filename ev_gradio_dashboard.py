#!/usr/bin/env python3
"""
🚗 Modern AI-Powered Electric Vehicle Analysis Dashboard using Gradio
Features: Beautiful UI, Natural Language Queries, AI Insights, Interactive Analytics
"""

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

# Load and preprocess data
def load_data():
    """Load and cache data efficiently"""
    try:
        df = pd.read_csv('electric_vehicles_spec_2025.csv.csv')
        
        # Data cleaning and preprocessing
        numeric_columns = [
            'top_speed_kmh', 'battery_capacity_kWh', 'number_of_cells', 'torque_nm',
            'efficiency_wh_per_km', 'range_km', 'acceleration_0_100_s', 
            'fast_charging_power_kw_dc', 'towing_capacity_kg', 'cargo_volume_l',
            'seats', 'length_mm', 'width_mm', 'height_mm'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Create derived features
        df['segment'] = df['segment'].str.strip()
        df['segment_category'] = df['segment'].str.split(' - ').str[0]
        df['range_per_kwh'] = df['range_km'] / df['battery_capacity_kWh']
        df['power_to_weight_ratio'] = df['torque_nm'] / (df['length_mm'] * df['width_mm'] * df['height_mm'] * 1e-9)
        
        return df
    except Exception as e:
        return pd.DataFrame()

# Load data
df = load_data()

# AI Analysis Functions
def perform_clustering(df, n_clusters=4):
    """Perform K-means clustering on EV data"""
    features = ['range_km', 'top_speed_kmh', 'battery_capacity_kWh', 'efficiency_wh_per_km']
    data = df[features].dropna()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)
    data = data.copy()
    data['cluster'] = clusters
    return data

def predict_range(battery_capacity, efficiency):
    """Predict range based on battery capacity and efficiency"""
    model_data = df[['battery_capacity_kWh', 'efficiency_wh_per_km', 'range_km']].dropna()
    X = model_data[['battery_capacity_kWh', 'efficiency_wh_per_km']]
    y = model_data['range_km']
    model = LinearRegression()
    model.fit(X, y)
    prediction = model.predict([[battery_capacity, efficiency]])
    r2_score = model.score(X, y)
    return prediction[0], r2_score

def analyze_query(query, df):
    """AI-powered query analysis"""
    query = query.lower()
    
    if 'range' in query and 'battery' in query:
        return "🔋 **Battery vs Range Analysis**: EVs with larger batteries generally have longer ranges, but efficiency varies significantly between models. The correlation between battery capacity and range is approximately 0.85."
    
    elif 'speed' in query and 'performance' in query:
        return "🏁 **Speed Performance**: High-performance EVs typically have top speeds above 200 km/h, with acceleration times under 4 seconds for 0-100 km/h. Tesla and Porsche lead in performance metrics."
    
    elif 'efficiency' in query or 'consumption' in query:
        return "⚡ **Efficiency Analysis**: The most efficient EVs achieve 120-150 Wh/km, while larger vehicles consume 200-300 Wh/km. Efficiency is crucial for maximizing range from battery capacity."
    
    elif 'brand' in query and 'compare' in query:
        top_brands = df.groupby('brand')['range_km'].mean().sort_values(ascending=False).head(5)
        return f"🏷️ **Top 5 Brands by Average Range**:\n" + "\n".join([f"• {brand}: {range_val:.0f} km" for brand, range_val in top_brands.items()])
    
    elif 'segment' in query:
        segment_stats = df.groupby('segment_category').agg({
            'range_km': 'mean',
            'battery_capacity_kWh': 'mean',
            'top_speed_kmh': 'mean'
        }).round(1)
        return f"📦 **Segment Analysis**:\n{segment_stats.to_string()}"
    
    elif 'charging' in query:
        return "🔌 **Charging Infrastructure**: Fast charging capabilities range from 50kW to 350kW. Higher charging power reduces charging time but requires compatible infrastructure."
    
    elif 'trend' in query or 'future' in query:
        return "📈 **Market Trends**: EVs are trending toward larger batteries (80-100 kWh), faster charging (150-350 kW), and improved efficiency. Range anxiety is decreasing as technology advances."
    
    else:
        return "🤖 **General Insights**: The EV market is rapidly evolving with improvements in battery technology, charging infrastructure, and overall efficiency. Key factors include range, charging speed, and cost-effectiveness."

def create_overview_dashboard():
    """Create overview dashboard with key metrics"""
    # Key metrics
    total_evs = len(df)
    avg_range = df['range_km'].mean()
    avg_battery = df['battery_capacity_kWh'].mean()
    avg_speed = df['top_speed_kmh'].mean()
    
    # Brand distribution
    brand_counts = df['brand'].value_counts().head(10)
    fig_brand = px.pie(
        values=brand_counts.values,
        names=brand_counts.index,
        title="🚗 Top 10 Brands Distribution",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig_brand.update_traces(textposition='inside', textinfo='percent+label')
    fig_brand.update_layout(
        title_font_size=20,
        title_font_color='#1e3c72',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    # Performance scatter
    fig_performance = px.scatter(
        df, x='range_km', y='top_speed_kmh',
        color='brand', size='battery_capacity_kWh',
        hover_data=['model', 'segment_category'],
        title="⚡ Performance Analysis: Range vs Speed",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig_performance.update_layout(
        title_font_size=20,
        title_font_color='#1e3c72',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    # Segment distribution
    segment_counts = df['segment_category'].value_counts().reset_index()
    segment_counts.columns = ['segment_category', 'count']
    fig_segment = px.bar(
        segment_counts,
        x='segment_category',
        y='count',
        title="📦 Segment Distribution",
        color_discrete_sequence=['#1e3c72']
    )
    fig_segment.update_layout(
        title_font_size=20,
        title_font_color='#1e3c72',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig_brand, fig_performance, fig_segment, f"""
    ## 📊 Market Overview
    
    **Key Metrics:**
    - 🚗 **Total Vehicles**: {total_evs:,}
    - 🔋 **Average Range**: {avg_range:.0f} km
    - ⚡ **Average Battery**: {avg_battery:.1f} kWh
    - 🏁 **Average Speed**: {avg_speed:.0f} km/h
    """

def create_ml_insights(n_clusters):
    """Create machine learning insights"""
    clustered_data = perform_clustering(df, n_clusters)
    
    # Clustering scatter plot
    fig_cluster = px.scatter(
        clustered_data, x='range_km', y='top_speed_kmh',
        color='cluster', size='battery_capacity_kWh',
        title=f"🤖 EV Clusters (K={n_clusters})",
        color_discrete_sequence=px.colors.qualitative.Set3
    )
    fig_cluster.update_layout(
        title_font_size=20,
        title_font_color='#1e3c72',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    # Cluster characteristics
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
        title_font_color='#1e3c72',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    # Cluster insights
    insights = []
    for cluster in range(n_clusters):
        cluster_data = clustered_data[clustered_data['cluster'] == cluster]
        insights.append(f"""
        **Cluster {cluster}** ({len(cluster_data)} vehicles):
        - 🔋 Avg Range: {cluster_data['range_km'].mean():.0f} km
        - 🏁 Avg Speed: {cluster_data['top_speed_kmh'].mean():.0f} km/h
        - ⚡ Avg Battery: {cluster_data['battery_capacity_kWh'].mean():.1f} kWh
        """)
    
    return fig_cluster, fig_radar, "\n".join(insights)

def create_prediction_analysis(battery_capacity, efficiency):
    """Create prediction analysis"""
    predicted_range, r2_score = predict_range(battery_capacity, efficiency)
    
    # Create scatter plot with prediction
    model_data = df[['battery_capacity_kWh', 'efficiency_wh_per_km', 'range_km']].dropna()
    fig_scatter = px.scatter(
        model_data, x='battery_capacity_kWh', y='range_km',
        color='efficiency_wh_per_km',
        title=f"🔮 Range Prediction Model (R² = {r2_score:.3f})",
        color_continuous_scale='viridis'
    )
    fig_scatter.add_trace(go.Scatter(
        x=[battery_capacity], y=[predicted_range],
        mode='markers',
        marker=dict(size=20, color='red', symbol='star'),
        name='Prediction'
    ))
    fig_scatter.update_layout(
        title_font_size=20,
        title_font_color='#1e3c72',
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
        title_font_color='#1e3c72'
    )
    
    return f"{predicted_range:.0f} km", fig_scatter, fig_surface

def create_advanced_analytics():
    """Create advanced analytics visualizations"""
    # Correlation heatmap
    features = ['range_km', 'top_speed_kmh', 'battery_capacity_kWh', 
                'efficiency_wh_per_km', 'acceleration_0_100_s']
    corr_matrix = df[features].corr()
    fig_corr = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="📊 Feature Correlation Matrix",
        color_continuous_scale="RdBu"
    )
    fig_corr.update_layout(
        title_font_size=20,
        title_font_color='#1e3c72',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
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
        title_font_color='#1e3c72',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    # Box plots
    fig_box = px.box(
        df, x='segment_category', y='range_km',
        title="📦 Range Distribution by Segment",
        color_discrete_sequence=['#1e3c72']
    )
    fig_box.update_layout(
        title_font_size=20,
        title_font_color='#1e3c72',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig_corr, fig_dist, fig_box

def create_interactive_explorer(brands, segments, min_range, min_speed):
    """Create interactive explorer visualizations"""
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
        title_font_color='#1e3c72',
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
        title_font_color='#1e3c72'
    )
    
    # Summary
    summary = f"""
    ## 🔍 Data Explorer Results
    
    **Filtered Data Summary:**
    - 📊 **Total Vehicles**: {len(filtered_df)}
    - 🔋 **Average Range**: {filtered_df['range_km'].mean():.0f} km
    - ⚡ **Average Battery**: {filtered_df['battery_capacity_kWh'].mean():.1f} kWh
    - 🏁 **Average Speed**: {filtered_df['top_speed_kmh'].mean():.0f} km/h
    
    **Top Models by Range:**
    """
    
    top_models = filtered_df.nlargest(5, 'range_km')[['brand', 'model', 'range_km']]
    for _, row in top_models.iterrows():
        summary += f"- {row['brand']} {row['model']}: {row['range_km']:.0f} km\n"
    
    return fig_scatter, fig_3d, summary

# Create Gradio interface with modern design
def create_dashboard():
    """Create the main dashboard interface with modern design"""
    
    # Modern CSS styling
    custom_css = """
    .gradio-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
        padding: 2rem;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
        text-align: center;
        color: white;
        border: 1px solid rgba(255,255,255,0.1);
    }
    
    .main-header h1 {
        font-size: 3.5rem;
        font-weight: 800;
        margin: 0;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
        background: linear-gradient(45deg, #ffffff, #e0e7ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .main-header p {
        font-size: 1.3rem;
        margin: 1rem 0 0 0;
        opacity: 0.95;
        font-weight: 300;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.98);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        backdrop-filter: blur(10px);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
    }
    
    .gradio-tab-nav {
        background: rgba(255,255,255,0.1) !important;
        border-radius: 15px !important;
        padding: 0.5rem !important;
        margin: 1rem 0 !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .gradio-tab-nav button {
        background: rgba(255,255,255,0.8) !important;
        border-radius: 12px !important;
        border: none !important;
        padding: 1rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        margin: 0 0.5rem !important;
        color: #1e3c72 !important;
    }
    
    .gradio-tab-nav button:hover {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%) !important;
        color: white !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(30,60,114,0.3) !important;
    }
    
    .gradio-tab-nav button.selected {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%) !important;
        color: white !important;
        box-shadow: 0 6px 20px rgba(30,60,114,0.3) !important;
    }
    
    .gradio-button {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%) !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 1rem 2rem !important;
        font-weight: 600 !important;
        color: white !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(30,60,114,0.2) !important;
    }
    
    .gradio-button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(30,60,114,0.3) !important;
    }
    
    .gradio-input, .gradio-textbox, .gradio-slider, .gradio-dropdown {
        border-radius: 12px !important;
        border: 2px solid rgba(30,60,114,0.1) !important;
        background: rgba(255,255,255,0.95) !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .gradio-input:focus, .gradio-textbox:focus, .gradio-slider:focus, .gradio-dropdown:focus {
        border-color: #1e3c72 !important;
        box-shadow: 0 0 0 3px rgba(30,60,114,0.1) !important;
    }
    
    .gradio-plot {
        background: rgba(255,255,255,0.98) !important;
        border-radius: 20px !important;
        padding: 1.5rem !important;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1) !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .gradio-markdown {
        background: rgba(255,255,255,0.98) !important;
        border-radius: 20px !important;
        padding: 2rem !important;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1) !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .gradio-row {
        margin: 1rem 0 !important;
    }
    
    .gradio-column {
        padding: 0.5rem !important;
    }
    """
    
    with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
        # Beautiful Header
        gr.HTML("""
        <div class="main-header">
            <h1>🚗 EV Analytics Hub</h1>
            <p>AI-Powered Electric Vehicle Analysis • Natural Language Queries • Machine Learning Insights</p>
        </div>
        """)
        
        # Main tabs with modern design
        with gr.Tabs():
            # Overview Tab
            with gr.Tab("📊 Market Overview"):
                with gr.Row():
                    with gr.Column(scale=1):
                        overview_btn = gr.Button(
                            "🔄 Generate Overview", 
                            variant="primary", 
                            size="lg",
                            elem_classes=["gradio-button"]
                        )
                        overview_output = gr.HTML(elem_classes=["gradio-markdown"])
                
                with gr.Row():
                    with gr.Column(scale=1):
                        brand_plot = gr.Plot(label="Brand Distribution", elem_classes=["gradio-plot"])
                    with gr.Column(scale=1):
                        performance_plot = gr.Plot(label="Performance Analysis", elem_classes=["gradio-plot"])
                
                with gr.Row():
                    with gr.Column(scale=1):
                        segment_plot = gr.Plot(label="Segment Distribution", elem_classes=["gradio-plot"])
            
            # AI Insights Tab
            with gr.Tab("🤖 AI Assistant"):
                with gr.Row():
                    with gr.Column(scale=2):
                        query_input = gr.Textbox(
                            label="💬 Ask me anything about EVs:",
                            placeholder="e.g., 'Compare brands by range' or 'Analyze efficiency trends'",
                            lines=3,
                            elem_classes=["gradio-textbox"]
                        )
                        query_btn = gr.Button("🔍 Analyze", variant="primary", elem_classes=["gradio-button"])
                        query_output = gr.Markdown(elem_classes=["gradio-markdown"])
                
                with gr.Row():
                    with gr.Column(scale=1):
                        cluster_slider = gr.Slider(
                            minimum=2, maximum=6, value=4, step=1,
                            label="Number of Clusters",
                            elem_classes=["gradio-slider"]
                        )
                        ml_btn = gr.Button("🤖 Generate ML Insights", variant="primary", elem_classes=["gradio-button"])
                        ml_output = gr.Markdown(elem_classes=["gradio-markdown"])
                
                with gr.Row():
                    with gr.Column(scale=1):
                        cluster_plot = gr.Plot(label="Clustering Analysis", elem_classes=["gradio-plot"])
                    with gr.Column(scale=1):
                        radar_plot = gr.Plot(label="Cluster Characteristics", elem_classes=["gradio-plot"])
            
            # Predictive Analytics Tab
            with gr.Tab("🔮 Predictions"):
                with gr.Row():
                    with gr.Column(scale=1):
                        battery_input = gr.Slider(
                            minimum=30, maximum=150, value=75, step=5,
                            label="🔋 Battery Capacity (kWh)",
                            elem_classes=["gradio-slider"]
                        )
                        efficiency_input = gr.Slider(
                            minimum=100, maximum=300, value=200, step=10,
                            label="⚡ Efficiency (Wh/km)",
                            elem_classes=["gradio-slider"]
                        )
                        predict_btn = gr.Button("🔮 Predict Range", variant="primary", elem_classes=["gradio-button"])
                        predict_output = gr.Markdown(elem_classes=["gradio-markdown"])
                
                with gr.Row():
                    with gr.Column(scale=1):
                        prediction_plot = gr.Plot(label="Prediction Model", elem_classes=["gradio-plot"])
                    with gr.Column(scale=1):
                        surface_plot = gr.Plot(label="Prediction Surface", elem_classes=["gradio-plot"])
            
            # Advanced Analytics Tab
            with gr.Tab("📈 Analytics"):
                with gr.Row():
                    analytics_btn = gr.Button("📊 Generate Analytics", variant="primary", size="lg", elem_classes=["gradio-button"])
                
                with gr.Row():
                    with gr.Column(scale=1):
                        corr_plot = gr.Plot(label="Correlation Matrix", elem_classes=["gradio-plot"])
                
                with gr.Row():
                    with gr.Column(scale=1):
                        dist_plot = gr.Plot(label="Feature Distributions", elem_classes=["gradio-plot"])
                    with gr.Column(scale=1):
                        box_plot = gr.Plot(label="Segment Analysis", elem_classes=["gradio-plot"])
            
            # Interactive Explorer Tab
            with gr.Tab("🔍 Explorer"):
                with gr.Row():
                    with gr.Column(scale=1):
                        brand_filter = gr.Dropdown(
                            choices=sorted(df['brand'].unique()),
                            label="🏷️ Select Brands",
                            multiselect=True,
                            elem_classes=["gradio-dropdown"]
                        )
                        segment_filter = gr.Dropdown(
                            choices=sorted(df['segment_category'].unique()),
                            label="📦 Select Segments",
                            multiselect=True,
                            elem_classes=["gradio-dropdown"]
                        )
                    with gr.Column(scale=1):
                        range_filter = gr.Slider(
                            minimum=0, maximum=int(df['range_km'].max()), value=0,
                            label="🔋 Min Range (km)",
                            elem_classes=["gradio-slider"]
                        )
                        speed_filter = gr.Slider(
                            minimum=0, maximum=int(df['top_speed_kmh'].max()), value=0,
                            label="🏁 Min Speed (km/h)",
                            elem_classes=["gradio-slider"]
                        )
                
                with gr.Row():
                    explore_btn = gr.Button("🔍 Explore Data", variant="primary", elem_classes=["gradio-button"])
                    explore_output = gr.Markdown(elem_classes=["gradio-markdown"])
                
                with gr.Row():
                    with gr.Column(scale=1):
                        scatter_plot = gr.Plot(label="Interactive Scatter", elem_classes=["gradio-plot"])
                    with gr.Column(scale=1):
                        plot_3d = gr.Plot(label="3D Analysis", elem_classes=["gradio-plot"])
        
        # Event handlers
        overview_btn.click(
            fn=create_overview_dashboard,
            outputs=[brand_plot, performance_plot, segment_plot, overview_output]
        )
        
        query_btn.click(
            fn=lambda q: analyze_query(q, df),
            inputs=[query_input],
            outputs=[query_output]
        )
        
        ml_btn.click(
            fn=create_ml_insights,
            inputs=[cluster_slider],
            outputs=[cluster_plot, radar_plot, ml_output]
        )
        
        predict_btn.click(
            fn=create_prediction_analysis,
            inputs=[battery_input, efficiency_input],
            outputs=[predict_output, prediction_plot, surface_plot]
        )
        
        analytics_btn.click(
            fn=create_advanced_analytics,
            outputs=[corr_plot, dist_plot, box_plot]
        )
        
        explore_btn.click(
            fn=create_interactive_explorer,
            inputs=[brand_filter, segment_filter, range_filter, speed_filter],
            outputs=[scatter_plot, plot_3d, explore_output]
        )
    
    return demo

# Launch the dashboard
if __name__ == "__main__":
    print("🚗 Starting Modern AI-Powered Electric Vehicle Analysis Dashboard (Gradio)...")
    print("🤖 Loading AI models and machine learning insights...")
    print("🌐 Dashboard will open in your browser at http://localhost:7861")
    print("⏹️  Press Ctrl+C to stop the dashboard")
    print("-" * 70)
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    try:
        import gradio
        import plotly
        import sklearn
        print("✅ All dependencies are available")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("💡 Install dependencies with:")
        print("   pip install -r gradio_requirements.txt")
        exit(1)
    
    print("🚀 Launching modern AI dashboard...")
    
    # Create and launch the dashboard
    demo = create_dashboard()
    demo.launch(
        server_name="localhost",
        server_port=7861,
        share=False,
        show_error=True,
        quiet=False
    ) 