#!/usr/bin/env python3
"""
Enhanced Electric Vehicle Analysis Dashboard using Streamlit
Features: Modern UI, Interactive Visualizations, Real-time Analytics
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime
import time

# Page configuration
st.set_page_config(
    page_title="🚗 EV Analytics Hub",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Modern gradient background */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }
    
    /* Enhanced header styling */
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        color: rgba(255,255,255,0.9);
        text-align: center;
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
    }
    
    /* Metric cards with glassmorphism */
    .metric-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.15);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e3c72;
        text-align: center;
        margin: 0;
    }
    
    .metric-label {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin: 0.5rem 0 0 0;
        font-weight: 500;
    }
    
    /* Enhanced sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.8);
        border-radius: 10px 10px 0 0;
        border: none;
        padding: 1rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        box-shadow: 0 4px 15px rgba(30,60,114,0.3);
    }
    
    /* Chart containers */
    .chart-container {
        background: rgba(255,255,255,0.95);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        border: 1px solid rgba(0,0,0,0.05);
    }
    
    /* Loading animation */
    .loading {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(255,255,255,.3);
        border-radius: 50%;
        border-top-color: #1e3c72;
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        .metric-value {
            font-size: 2rem;
        }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)  # Cache for 1 hour
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
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()

def create_metric_card(value, label, icon="📊"):
    """Create a styled metric card"""
    return f"""
    <div class="metric-card">
        <div class="metric-value">{icon} {value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """

def main():
    # Header with modern design
    st.markdown("""
    <div class="main-header">
        <h1>🚗 Electric Vehicle Analytics Hub</h1>
        <p>Comprehensive Analysis & Insights for 2025 EV Market</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data with loading indicator
    with st.spinner("🔄 Loading EV data..."):
        df = load_data()
    
    if df.empty:
        st.error("❌ Failed to load data. Please check if the CSV file exists.")
        return
    
    # Sidebar with enhanced styling
    st.sidebar.markdown("## 🎛️ Dashboard Controls")
    st.sidebar.markdown("---")
    
    # Filter options
    selected_brands = st.sidebar.multiselect(
        "🏷️ Select Brands",
        options=sorted(df['brand'].unique()),
        default=sorted(df['brand'].unique())[:5]
    )
    
    selected_segments = st.sidebar.multiselect(
        "📦 Select Segments",
        options=sorted(df['segment_category'].unique()),
        default=sorted(df['segment_category'].unique())
    )
    
    range_filter = st.sidebar.slider(
        "🔋 Min Range (km)",
        min_value=0,
        max_value=int(df['range_km'].max()),
        value=(0, int(df['range_km'].max())),
        step=10
    )
    
    # Filter data
    filtered_df = df[
        (df['brand'].isin(selected_brands)) &
        (df['segment_category'].isin(selected_segments)) &
        (df['range_km'] >= range_filter[0]) &
        (df['range_km'] <= range_filter[1])
    ]
    
    # Main content with tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Overview", "🏷️ Brands", "⚡ Performance", "🔋 Battery & Range", 
        "⚡ Charging", "📦 Segments", "📈 Trends", "🔍 Explorer"
    ])
    
    with tab1:
        st.markdown("## 📊 Market Overview")
        
        # Key metrics in a grid
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(create_metric_card(
                f"{len(filtered_df):,}", 
                "Total Vehicles", 
                "🚗"
            ), unsafe_allow_html=True)
        
        with col2:
            st.markdown(create_metric_card(
                f"{filtered_df['range_km'].mean():.0f} km", 
                "Avg Range", 
                "🔋"
            ), unsafe_allow_html=True)
        
        with col3:
            st.markdown(create_metric_card(
                f"{filtered_df['battery_capacity_kWh'].mean():.1f} kWh", 
                "Avg Battery", 
                "⚡"
            ), unsafe_allow_html=True)
        
        with col4:
            st.markdown(create_metric_card(
                f"{filtered_df['top_speed_kmh'].mean():.0f} km/h", 
                "Avg Speed", 
                "🏁"
            ), unsafe_allow_html=True)
        
        # Market distribution charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig_brand_dist = px.pie(
                filtered_df, 
                names='brand', 
                title="🚗 Brand Distribution",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_brand_dist.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_brand_dist, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            segment_counts = filtered_df['segment_category'].value_counts().reset_index()
            segment_counts.columns = ['segment_category', 'count']
            fig_segment_dist = px.bar(
                segment_counts,
                x='segment_category',
                y='count',
                title="📦 Segment Distribution",
                color_discrete_sequence=['#1e3c72']
            )
            fig_segment_dist.update_layout(xaxis_title="Segment", yaxis_title="Count")
            st.plotly_chart(fig_segment_dist, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Performance scatter plot
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig_performance = px.scatter(
            filtered_df,
            x='range_km',
            y='top_speed_kmh',
            color='brand',
            size='battery_capacity_kWh',
            hover_data=['model', 'segment_category'],
            title="⚡ Performance Analysis: Range vs Speed",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        fig_performance.update_layout(
            xaxis_title="Range (km)",
            yaxis_title="Top Speed (km/h)",
            height=500
        )
        st.plotly_chart(fig_performance, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown("## 🏷️ Brand Analysis")
        
        # Brand performance comparison
        brand_stats = filtered_df.groupby('brand').agg({
            'range_km': ['mean', 'count'],
            'battery_capacity_kWh': 'mean',
            'top_speed_kmh': 'mean',
            'efficiency_wh_per_km': 'mean'
        }).round(2)
        
        brand_stats.columns = ['Avg Range', 'Count', 'Avg Battery', 'Avg Speed', 'Avg Efficiency']
        brand_stats = brand_stats.sort_values('Count', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig_brand_range = px.bar(
                brand_stats.reset_index(),
                x='brand',
                y='Avg Range',
                title="🔋 Average Range by Brand",
                color='Count',
                color_continuous_scale='viridis'
            )
            fig_brand_range.update_layout(xaxis_title="Brand", yaxis_title="Average Range (km)")
            st.plotly_chart(fig_brand_range, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig_brand_efficiency = px.bar(
                brand_stats.reset_index(),
                x='brand',
                y='Avg Efficiency',
                title="⚡ Average Efficiency by Brand",
                color='Avg Speed',
                color_continuous_scale='plasma'
            )
            fig_brand_efficiency.update_layout(xaxis_title="Brand", yaxis_title="Average Efficiency (Wh/km)")
            st.plotly_chart(fig_brand_efficiency, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Brand statistics table
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("### 📊 Brand Statistics")
        st.dataframe(brand_stats, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        st.markdown("## ⚡ Performance Analysis")
        
        # Performance metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(create_metric_card(
                f"{filtered_df['acceleration_0_100_s'].mean():.1f}s", 
                "Avg 0-100 km/h", 
                "🏁"
            ), unsafe_allow_html=True)
        
        with col2:
            st.markdown(create_metric_card(
                f"{filtered_df['torque_nm'].mean():.0f} Nm", 
                "Avg Torque", 
                "🔧"
            ), unsafe_allow_html=True)
        
        with col3:
            st.markdown(create_metric_card(
                f"{filtered_df['power_to_weight_ratio'].mean():.2f}", 
                "Power/Weight Ratio", 
                "⚖️"
            ), unsafe_allow_html=True)
        
        # Performance visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig_acceleration = px.histogram(
                filtered_df,
                x='acceleration_0_100_s',
                nbins=20,
                title="🏁 Acceleration Distribution",
                color_discrete_sequence=['#1e3c72']
            )
            fig_acceleration.update_layout(xaxis_title="0-100 km/h Time (s)", yaxis_title="Count")
            st.plotly_chart(fig_acceleration, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig_torque = px.scatter(
                filtered_df,
                x='torque_nm',
                y='acceleration_0_100_s',
                color='brand',
                size='battery_capacity_kWh',
                title="🔧 Torque vs Acceleration",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_torque.update_layout(xaxis_title="Torque (Nm)", yaxis_title="0-100 km/h Time (s)")
            st.plotly_chart(fig_torque, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        st.markdown("## 🔋 Battery & Range Analysis")
        
        # Battery metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(create_metric_card(
                f"{filtered_df['battery_capacity_kWh'].max():.0f} kWh", 
                "Largest Battery", 
                "🔋"
            ), unsafe_allow_html=True)
        
        with col2:
            st.markdown(create_metric_card(
                f"{filtered_df['range_km'].max():.0f} km", 
                "Longest Range", 
                "🛣️"
            ), unsafe_allow_html=True)
        
        with col3:
            st.markdown(create_metric_card(
                f"{filtered_df['range_per_kwh'].mean():.1f} km/kWh", 
                "Avg Efficiency", 
                "⚡"
            ), unsafe_allow_html=True)
        
        # Battery visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig_battery_range = px.scatter(
                filtered_df,
                x='battery_capacity_kWh',
                y='range_km',
                color='brand',
                size='efficiency_wh_per_km',
                hover_data=['model', 'segment_category'],
                title="🔋 Battery Capacity vs Range",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_battery_range.update_layout(
                xaxis_title="Battery Capacity (kWh)",
                yaxis_title="Range (km)"
            )
            st.plotly_chart(fig_battery_range, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig_efficiency = px.box(
                filtered_df,
                x='segment_category',
                y='range_per_kwh',
                title="⚡ Range Efficiency by Segment",
                color_discrete_sequence=['#1e3c72']
            )
            fig_efficiency.update_layout(xaxis_title="Segment", yaxis_title="Range per kWh (km/kWh)")
            st.plotly_chart(fig_efficiency, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab5:
        st.markdown("## ⚡ Charging Infrastructure")
        
        # Charging metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(create_metric_card(
                f"{filtered_df['fast_charging_power_kw_dc'].max():.0f} kW", 
                "Max Fast Charging", 
                "⚡"
            ), unsafe_allow_html=True)
        
        with col2:
            st.markdown(create_metric_card(
                f"{filtered_df['fast_charging_power_kw_dc'].mean():.0f} kW", 
                "Avg Fast Charging", 
                "🔌"
            ), unsafe_allow_html=True)
        
        with col3:
            st.markdown(create_metric_card(
                f"{len(filtered_df[filtered_df['fast_charging_power_kw_dc'] > 100])}", 
                "100+ kW Charging", 
                "🚀"
            ), unsafe_allow_html=True)
        
        # Charging visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig_charging = px.histogram(
                filtered_df,
                x='fast_charging_power_kw_dc',
                nbins=20,
                title="⚡ Fast Charging Power Distribution",
                color_discrete_sequence=['#1e3c72']
            )
            fig_charging.update_layout(xaxis_title="Fast Charging Power (kW)", yaxis_title="Count")
            st.plotly_chart(fig_charging, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig_charging_brand = px.box(
                filtered_df,
                x='brand',
                y='fast_charging_power_kw_dc',
                title="🔌 Fast Charging by Brand",
                color_discrete_sequence=['#1e3c72']
            )
            fig_charging_brand.update_layout(xaxis_title="Brand", yaxis_title="Fast Charging Power (kW)")
            st.plotly_chart(fig_charging_brand, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab6:
        st.markdown("## 📦 Market Segments")
        
        # Segment analysis
        segment_stats = filtered_df.groupby('segment_category').agg({
            'range_km': ['mean', 'count'],
            'battery_capacity_kWh': 'mean',
            'top_speed_kmh': 'mean',
            'fast_charging_power_kw_dc': 'mean'
        }).round(2)
        
        segment_stats.columns = ['Avg Range', 'Count', 'Avg Battery', 'Avg Speed', 'Avg Charging']
        segment_stats = segment_stats.sort_values('Count', ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig_segment_range = px.bar(
                segment_stats.reset_index(),
                x='segment_category',
                y='Avg Range',
                title="🔋 Average Range by Segment",
                color='Count',
                color_continuous_scale='viridis'
            )
            fig_segment_range.update_layout(xaxis_title="Segment", yaxis_title="Average Range (km)")
            st.plotly_chart(fig_segment_range, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig_segment_speed = px.bar(
                segment_stats.reset_index(),
                x='segment_category',
                y='Avg Speed',
                title="🏁 Average Speed by Segment",
                color='Avg Battery',
                color_continuous_scale='plasma'
            )
            fig_segment_speed.update_layout(xaxis_title="Segment", yaxis_title="Average Speed (km/h)")
            st.plotly_chart(fig_segment_speed, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Segment statistics
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("### 📊 Segment Statistics")
        st.dataframe(segment_stats, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab7:
        st.markdown("## 📈 Market Trends")
        
        # Trend analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig_range_trend = px.scatter(
                filtered_df,
                x='battery_capacity_kWh',
                y='range_km',
                color='segment_category',
                size='top_speed_kmh',
                title="📈 Battery vs Range Trend",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_range_trend.update_layout(
                xaxis_title="Battery Capacity (kWh)",
                yaxis_title="Range (km)"
            )
            st.plotly_chart(fig_range_trend, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            fig_efficiency_trend = px.scatter(
                filtered_df,
                x='efficiency_wh_per_km',
                y='range_per_kwh',
                color='brand',
                size='battery_capacity_kWh',
                title="⚡ Efficiency vs Range per kWh",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_efficiency_trend.update_layout(
                xaxis_title="Efficiency (Wh/km)",
                yaxis_title="Range per kWh (km/kWh)"
            )
            st.plotly_chart(fig_efficiency_trend, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # 3D visualization
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        fig_3d = px.scatter_3d(
            filtered_df,
            x='battery_capacity_kWh',
            y='range_km',
            z='top_speed_kmh',
            color='segment_category',
            size='efficiency_wh_per_km',
            hover_data=['brand', 'model'],
            title="🌐 3D Analysis: Battery vs Range vs Speed"
        )
        fig_3d.update_layout(height=600)
        st.plotly_chart(fig_3d, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab8:
        st.markdown("## 🔍 Interactive Data Explorer")
        
        # Advanced filters
        col1, col2 = st.columns(2)
        
        with col1:
            min_speed = st.slider("🏁 Min Speed (km/h)", 0, 350, 0)
            min_acceleration = st.slider("⚡ Max 0-100 time (s)", 0, 20, 20)
        
        with col2:
            min_charging = st.slider("🔌 Min Fast Charging (kW)", 0, 350, 0)
            min_torque = st.slider("🔧 Min Torque (Nm)", 0, 1000, 0)
        
        # Apply additional filters
        explorer_df = filtered_df[
            (filtered_df['top_speed_kmh'] >= min_speed) &
            (filtered_df['acceleration_0_100_s'] <= min_acceleration) &
            (filtered_df['fast_charging_power_kw_dc'] >= min_charging) &
            (filtered_df['torque_nm'] >= min_torque)
        ]
        
        # Results summary
        st.markdown(f"### 📊 Found {len(explorer_df)} vehicles matching criteria")
        
        # Interactive scatter plot
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        x_axis = st.selectbox("X-axis", ['range_km', 'battery_capacity_kWh', 'top_speed_kmh', 'efficiency_wh_per_km'])
        y_axis = st.selectbox("Y-axis", ['top_speed_kmh', 'range_km', 'battery_capacity_kWh', 'efficiency_wh_per_km'])
        
        fig_explorer = px.scatter(
            explorer_df,
            x=x_axis,
            y=y_axis,
            color='brand',
            size='battery_capacity_kWh',
            hover_data=['model', 'segment_category'],
            title=f"🔍 Interactive Explorer: {x_axis} vs {y_axis}",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_explorer, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Data table
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("### 📋 Detailed Results")
        display_columns = ['brand', 'model', 'range_km', 'battery_capacity_kWh', 
                          'top_speed_kmh', 'acceleration_0_100_s', 'fast_charging_power_kw_dc']
        st.dataframe(explorer_df[display_columns].sort_values('range_km', ascending=False), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🚗 Electric Vehicle Analytics Hub | Built with Streamlit & Plotly</p>
        <p>Data Source: Electric Vehicles Specification 2025</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 