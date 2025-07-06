#!/usr/bin/env python3
"""
Interactive Electric Vehicle Analysis Dashboard
Streamlit app for exploring EV market data 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="EV Market Analysis 2025",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and preprocess the electric vehicle dataset"""
    df = pd.read_csv('electric_vehicles_spec_2025.csv.csv')
    
    # Data cleaning
    numeric_columns = [
        'top_speed_kmh', 'battery_capacity_kWh', 'number_of_cells', 'torque_nm',
        'efficiency_wh_per_km', 'range_km', 'acceleration_0_100_s', 
        'fast_charging_power_kw_dc', 'towing_capacity_kg', 'cargo_volume_l',
        'seats', 'length_mm', 'width_mm', 'height_mm'
    ]
    
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Clean segment column
    df['segment'] = df['segment'].str.strip()
    df['segment_category'] = df['segment'].str.split(' - ').str[0]
    
    # Add calculated columns
    df['range_per_kwh'] = df['range_km'] / df['battery_capacity_kWh']
    df['power_to_weight_ratio'] = df['torque_nm'] / (df['length_mm'] * df['width_mm'] * df['height_mm'] * 1e-9)
    
    return df

def main():
    """Main dashboard function"""
    
    # Load data
    df = load_data()
    
    # Sidebar navigation
    st.sidebar.title("🚗 EV Analysis Dashboard")
    page = st.sidebar.selectbox(
        "Choose Analysis Section:",
        ["📊 Overview", "🏭 Brand Analysis", "⚡ Performance", "🔋 Battery & Range", 
         "⚡ Charging", "🚙 Segments", "📈 Trends", "🔍 Data Explorer"]
    )
    
    # Header
    st.markdown('<h1 class="main-header">Electric Vehicle Market Analysis 2025</h1>', unsafe_allow_html=True)
    
    # Page routing
    if page == "📊 Overview":
        show_overview(df)
    elif page == "🏭 Brand Analysis":
        show_brand_analysis(df)
    elif page == "⚡ Performance":
        show_performance_analysis(df)
    elif page == "🔋 Battery & Range":
        show_battery_analysis(df)
    elif page == "⚡ Charging":
        show_charging_analysis(df)
    elif page == "🚙 Segments":
        show_segment_analysis(df)
    elif page == "📈 Trends":
        show_trends_analysis(df)
    elif page == "🔍 Data Explorer":
        show_data_explorer(df)

def show_overview(df):
    """Overview page with key metrics and insights"""
    st.header("📊 Market Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total EV Models", f"{len(df):,}")
    
    with col2:
        st.metric("Number of Brands", f"{df['brand'].nunique()}")
    
    with col3:
        avg_range = df['range_km'].mean()
        st.metric("Average Range", f"{avg_range:.0f} km")
    
    with col4:
        avg_battery = df['battery_capacity_kWh'].mean()
        st.metric("Average Battery", f"{avg_battery:.1f} kWh")
    
    # Top performers
    st.subheader("🏆 Top Performers")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Longest Range:**")
        longest_range = df.loc[df['range_km'].idxmax()]
        st.info(f"{longest_range['brand']} {longest_range['model']} - {longest_range['range_km']:.0f} km")
        
        st.markdown("**Fastest Speed:**")
        fastest = df.loc[df['top_speed_kmh'].idxmax()]
        st.info(f"{fastest['brand']} {fastest['model']} - {fastest['top_speed_kmh']:.0f} km/h")
    
    with col2:
        st.markdown("**Quickest Acceleration:**")
        quickest = df.loc[df['acceleration_0_100_s'].idxmin()]
        st.info(f"{quickest['brand']} {quickest['model']} - {quickest['acceleration_0_100_s']:.1f}s")
        
        st.markdown("**Largest Battery:**")
        largest_battery = df.loc[df['battery_capacity_kWh'].idxmax()]
        st.info(f"{largest_battery['brand']} {largest_battery['model']} - {largest_battery['battery_capacity_kWh']:.1f} kWh")
    
    # Interactive scatter plot
    st.subheader("🎯 Range vs Speed Analysis")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        selected_brands = st.multiselect(
            "Select Brands:",
            options=sorted(df['brand'].unique()),
            default=sorted(df['brand'].unique())[:5]
        )
    
    with col2:
        selected_segments = st.multiselect(
            "Select Segments:",
            options=sorted(df['segment_category'].unique()),
            default=sorted(df['segment_category'].unique())
        )
    
    with col3:
        min_range = st.slider("Min Range (km)", 0, int(df['range_km'].max()), 0)
    
    # Filter data
    filtered_df = df[
        (df['brand'].isin(selected_brands)) &
        (df['segment_category'].isin(selected_segments)) &
        (df['range_km'] >= min_range)
    ]
    
    # Create interactive scatter plot
    fig = px.scatter(
        filtered_df, 
        x='top_speed_kmh', 
        y='range_km',
        color='brand',
        size='battery_capacity_kWh',
        hover_data=['model', 'acceleration_0_100_s', 'efficiency_wh_per_km'],
        title="Range vs Speed by Brand (Size = Battery Capacity)",
        labels={'top_speed_kmh': 'Top Speed (km/h)', 'range_km': 'Range (km)'}
    )
    
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)

def show_brand_analysis(df):
    """Brand analysis page"""
    st.header("🏭 Brand Analysis")
    
    # Brand distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Models by Brand")
        brand_counts = df['brand'].value_counts().head(15)
        
        fig = px.bar(
            x=brand_counts.values,
            y=brand_counts.index,
            orientation='h',
            title="Top 15 Brands by Number of Models",
            labels={'x': 'Number of Models', 'y': 'Brand'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🏆 Average Range by Brand")
        brand_range = df.groupby('brand')['range_km'].agg(['mean', 'count']).reset_index()
        brand_range = brand_range[brand_range['count'] >= 2].sort_values('mean', ascending=False)
        
        fig = px.bar(
            brand_range.head(15),
            x='mean',
            y='brand',
            orientation='h',
            title="Average Range by Brand (Min. 2 Models)",
            labels={'mean': 'Average Range (km)', 'brand': 'Brand'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Brand comparison
    st.subheader("🔍 Brand Comparison")
    
    selected_brands = st.multiselect(
        "Select brands to compare:",
        options=sorted(df['brand'].unique()),
        default=sorted(df['brand'].unique())[:5]
    )
    
    if selected_brands:
        brand_comparison = df[df['brand'].isin(selected_brands)].groupby('brand').agg({
            'range_km': 'mean',
            'top_speed_kmh': 'mean',
            'battery_capacity_kWh': 'mean',
            'acceleration_0_100_s': 'mean',
            'efficiency_wh_per_km': 'mean'
        }).round(1)
        
        st.dataframe(brand_comparison, use_container_width=True)
        
        # Radar chart for selected brands
        if len(selected_brands) > 0:
            fig = go.Figure()
            
            for brand in selected_brands:
                brand_data = brand_comparison.loc[brand]
                fig.add_trace(go.Scatterpolar(
                    r=[brand_data['range_km'], brand_data['top_speed_kmh'], 
                       brand_data['battery_capacity_kWh'], brand_data['acceleration_0_100_s'],
                       brand_data['efficiency_wh_per_km']],
                    theta=['Range', 'Speed', 'Battery', 'Acceleration', 'Efficiency'],
                    fill='toself',
                    name=brand
                ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, brand_comparison.max().max()])),
                showlegend=True,
                title="Brand Performance Comparison"
            )
            st.plotly_chart(fig, use_container_width=True)

def show_performance_analysis(df):
    """Performance analysis page"""
    st.header("⚡ Performance Analysis")
    
    # Performance correlation
    st.subheader("📈 Performance Correlations")
    
    performance_cols = ['top_speed_kmh', 'range_km', 'acceleration_0_100_s', 
                       'battery_capacity_kWh', 'efficiency_wh_per_km']
    
    corr_matrix = df[performance_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Performance Metrics Correlation Matrix",
        color_continuous_scale="RdBu"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Speed vs Range by segment
    st.subheader("🎯 Speed vs Range by Segment")
    
    fig = px.scatter(
        df,
        x='top_speed_kmh',
        y='range_km',
        color='segment_category',
        hover_data=['brand', 'model'],
        title="Speed vs Range by Vehicle Segment"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Top performers
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏁 Fastest EVs")
        fastest_evs = df.nlargest(10, 'top_speed_kmh')[['brand', 'model', 'top_speed_kmh']]
        fig = px.bar(
            fastest_evs,
            x='top_speed_kmh',
            y=[f"{row['brand']} {row['model']}" for _, row in fastest_evs.iterrows()],
            orientation='h',
            title="Top 10 Fastest Electric Vehicles"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("⚡ Quickest Acceleration")
        quickest_evs = df.nsmallest(10, 'acceleration_0_100_s')[['brand', 'model', 'acceleration_0_100_s']]
        fig = px.bar(
            quickest_evs,
            x='acceleration_0_100_s',
            y=[f"{row['brand']} {row['model']}" for _, row in quickest_evs.iterrows()],
            orientation='h',
            title="Top 10 Quickest EVs (0-100 km/h)"
        )
        st.plotly_chart(fig, use_container_width=True)

def show_battery_analysis(df):
    """Battery and range analysis page"""
    st.header("🔋 Battery & Range Analysis")
    
    # Battery capacity distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Battery Capacity Distribution")
        fig = px.histogram(
            df,
            x='battery_capacity_kWh',
            nbins=30,
            title="Distribution of Battery Capacities"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Range Distribution")
        fig = px.histogram(
            df,
            x='range_km',
            nbins=30,
            title="Distribution of Ranges"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Range vs Battery Capacity
    st.subheader("🎯 Range vs Battery Capacity")
    
    # Add efficiency calculation
    df['range_per_kwh'] = df['range_km'] / df['battery_capacity_kWh']
    
    fig = px.scatter(
        df,
        x='battery_capacity_kWh',
        y='range_km',
        color='range_per_kwh',
        size='efficiency_wh_per_km',
        hover_data=['brand', 'model'],
        title="Range vs Battery Capacity (Color: Efficiency, Size: Energy Consumption)",
        labels={'range_per_kwh': 'Range per kWh (km/kWh)'}
    )
    
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    # Most efficient EVs
    st.subheader("🌱 Most Efficient EVs")
    efficient_evs = df.nlargest(10, 'range_per_kwh')[['brand', 'model', 'range_km', 'battery_capacity_kWh', 'range_per_kwh']]
    st.dataframe(efficient_evs.round(2), use_container_width=True)

def show_charging_analysis(df):
    """Charging infrastructure analysis page"""
    st.header("⚡ Charging Infrastructure Analysis")
    
    # Fast charging power distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Fast Charging Power Distribution")
        fig = px.histogram(
            df,
            x='fast_charging_power_kw_dc',
            nbins=20,
            title="Distribution of Fast Charging Power"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🔌 Charging Port Types")
        port_counts = df['fast_charge_port'].value_counts()
        fig = px.pie(
            values=port_counts.values,
            names=port_counts.index,
            title="Fast Charging Port Types"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Fastest charging EVs
    st.subheader("⚡ Fastest Charging EVs")
    fastest_charging = df.nlargest(10, 'fast_charging_power_kw_dc')[['brand', 'model', 'fast_charging_power_kw_dc', 'battery_capacity_kWh']]
    
    fig = px.bar(
        fastest_charging,
        x='fast_charging_power_kw_dc',
        y=[f"{row['brand']} {row['model']}" for _, row in fastest_charging.iterrows()],
        orientation='h',
        title="Top 10 Fastest Charging Electric Vehicles"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Charging vs Battery relationship
    st.subheader("🔋 Charging Power vs Battery Capacity")
    fig = px.scatter(
        df,
        x='battery_capacity_kWh',
        y='fast_charging_power_kw_dc',
        color='brand',
        hover_data=['model'],
        title="Charging Power vs Battery Capacity by Brand"
    )
    st.plotly_chart(fig, use_container_width=True)

def show_segment_analysis(df):
    """Vehicle segments analysis page"""
    st.header("🚙 Vehicle Segments Analysis")
    
    # Segment distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Distribution by Segment")
        segment_counts = df['segment_category'].value_counts()
        fig = px.pie(
            values=segment_counts.values,
            names=segment_counts.index,
            title="Distribution by Vehicle Segment"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🚗 Body Types")
        body_counts = df['car_body_type'].value_counts().head(10)
        fig = px.bar(
            x=body_counts.values,
            y=body_counts.index,
            orientation='h',
            title="Top 10 Body Types"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance by segment
    st.subheader("📈 Performance by Segment")
    
    segment_performance = df.groupby('segment_category').agg({
        'top_speed_kmh': 'mean',
        'range_km': 'mean',
        'battery_capacity_kWh': 'mean',
        'acceleration_0_100_s': 'mean'
    }).round(1)
    
    fig = px.bar(
        segment_performance,
        barmode='group',
        title="Average Performance by Vehicle Segment"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Segment comparison table
    st.subheader("📊 Segment Comparison")
    st.dataframe(segment_performance, use_container_width=True)

def show_trends_analysis(df):
    """Market trends analysis page"""
    st.header("📈 Market Trends Analysis")
    
    # Market trends overview
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Average Range", f"{df['range_km'].mean():.0f} km")
    
    with col2:
        st.metric("Average Battery", f"{df['battery_capacity_kWh'].mean():.1f} kWh")
    
    with col3:
        st.metric("Average Speed", f"{df['top_speed_kmh'].mean():.0f} km/h")
    
    with col4:
        st.metric("Average Efficiency", f"{df['efficiency_wh_per_km'].mean():.0f} Wh/km")
    
    # Distribution plots
    st.subheader("📊 Market Distributions")
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Range Distribution', 'Efficiency Distribution', 
                       'Acceleration Distribution', 'Battery vs Range'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Range distribution
    fig.add_trace(
        go.Histogram(x=df['range_km'].dropna(), name="Range"),
        row=1, col=1
    )
    
    # Efficiency distribution
    fig.add_trace(
        go.Histogram(x=df['efficiency_wh_per_km'].dropna(), name="Efficiency"),
        row=1, col=2
    )
    
    # Acceleration distribution
    fig.add_trace(
        go.Histogram(x=df['acceleration_0_100_s'].dropna(), name="Acceleration"),
        row=2, col=1
    )
    
    # Battery vs Range scatter
    fig.add_trace(
        go.Scatter(x=df['battery_capacity_kWh'], y=df['range_km'], 
                  mode='markers', name="Battery vs Range"),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    
    # Market insights
    st.subheader("💡 Market Insights")
    
    insights = f"""
    ### Key Findings:
    
    1. **Market Size**: {len(df)} electric vehicle models available in 2025
    2. **Brand Diversity**: {df['brand'].nunique()} different manufacturers
    3. **Range Performance**: Average range of {df['range_km'].mean():.0f} km
    4. **Battery Technology**: Average battery capacity of {df['battery_capacity_kWh'].mean():.1f} kWh
    5. **Performance**: Average top speed of {df['top_speed_kmh'].mean():.0f} km/h
    6. **Efficiency**: Average energy consumption of {df['efficiency_wh_per_km'].mean():.0f} Wh/km
    
    ### Market Trends:
    - Battery capacities continue to increase
    - Fast charging capabilities improving
    - Chinese manufacturers gaining market share
    - Luxury segment pushing performance boundaries
    """
    
    st.markdown(insights)

def show_data_explorer(df):
    """Interactive data explorer page"""
    st.header("🔍 Data Explorer")
    
    # Filters
    st.subheader("🔧 Filters")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_brands = st.multiselect(
            "Brand:",
            options=sorted(df['brand'].unique()),
            default=[]
        )
    
    with col2:
        selected_segments = st.multiselect(
            "Segment:",
            options=sorted(df['segment_category'].unique()),
            default=[]
        )
    
    with col3:
        selected_body_types = st.multiselect(
            "Body Type:",
            options=sorted(df['car_body_type'].unique()),
            default=[]
        )
    
    # Range filters
    col1, col2 = st.columns(2)
    
    with col1:
        range_min, range_max = st.slider(
            "Range (km):",
            min_value=float(df['range_km'].min()),
            max_value=float(df['range_km'].max()),
            value=(float(df['range_km'].min()), float(df['range_km'].max()))
        )
    
    with col2:
        speed_min, speed_max = st.slider(
            "Top Speed (km/h):",
            min_value=float(df['top_speed_kmh'].min()),
            max_value=float(df['top_speed_kmh'].max()),
            value=(float(df['top_speed_kmh'].min()), float(df['top_speed_kmh'].max()))
        )
    
    # Apply filters
    filtered_df = df.copy()
    
    if selected_brands:
        filtered_df = filtered_df[filtered_df['brand'].isin(selected_brands)]
    
    if selected_segments:
        filtered_df = filtered_df[filtered_df['segment_category'].isin(selected_segments)]
    
    if selected_body_types:
        filtered_df = filtered_df[filtered_df['car_body_type'].isin(selected_body_types)]
    
    filtered_df = filtered_df[
        (filtered_df['range_km'] >= range_min) & (filtered_df['range_km'] <= range_max) &
        (filtered_df['top_speed_kmh'] >= speed_min) & (filtered_df['top_speed_kmh'] <= speed_max)
    ]
    
    # Display results
    st.subheader(f"📊 Results ({len(filtered_df)} vehicles found)")
    
    # Summary statistics
    if len(filtered_df) > 0:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Avg Range", f"{filtered_df['range_km'].mean():.0f} km")
        
        with col2:
            st.metric("Avg Speed", f"{filtered_df['top_speed_kmh'].mean():.0f} km/h")
        
        with col3:
            st.metric("Avg Battery", f"{filtered_df['battery_capacity_kWh'].mean():.1f} kWh")
        
        with col4:
            st.metric("Avg Efficiency", f"{filtered_df['efficiency_wh_per_km'].mean():.0f} Wh/km")
        
        # Data table
        st.subheader("📋 Vehicle Details")
        
        # Select columns to display
        columns_to_show = st.multiselect(
            "Select columns to display:",
            options=df.columns.tolist(),
            default=['brand', 'model', 'range_km', 'top_speed_kmh', 'battery_capacity_kWh', 'segment_category']
        )
        
        if columns_to_show:
            st.dataframe(filtered_df[columns_to_show], use_container_width=True)
        
        # Download filtered data
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download filtered data as CSV",
            data=csv,
            file_name="filtered_ev_data.csv",
            mime="text/csv"
        )
    else:
        st.warning("No vehicles match the selected criteria. Try adjusting the filters.")

if __name__ == "__main__":
    main() 