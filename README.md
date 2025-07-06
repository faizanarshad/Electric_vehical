# Electric Vehicle Market Analysis 2025

## 📊 Project Overview

This project provides a comprehensive analysis of the electric vehicle market for 2025, exploring trends, performance metrics, and market insights across different manufacturers and vehicle segments.

## 🚗 Dataset Information

- **Source**: EV Database 2025
- **Records**: 478 electric vehicle models
- **Features**: 22 columns including performance, battery, and specification data
- **Coverage**: Global market with focus on major manufacturers

### Key Features Analyzed:
- **Performance**: Top speed, acceleration, range
- **Battery**: Capacity, efficiency, charging capabilities
- **Specifications**: Dimensions, cargo space, seating
- **Market**: Brand analysis, segments, body types

## 📁 Files Included

### Analysis Files:
- `electric_vehicle_analysis_2025.ipynb` - Complete Jupyter notebook with analysis
- `electric_vehicles_spec_2025.csv.csv` - Original dataset
- `electric_vehicles_analyzed_2025.csv` - Cleaned and analyzed dataset
- `ev_summary_statistics_2025.csv` - Summary statistics

### 🚗 Interactive Dashboards:
- **Streamlit Dashboard**: `ev_dashboard.py` - Traditional interactive dashboard
- **Dash Dashboard**: `ev_dash_dashboard.py` - Advanced ML-powered dashboard
- **Gradio Dashboard**: `ev_gradio_dashboard.py` - AI-focused modern interface
- **Launcher Scripts**: `run_dashboard.py`, `run_dash_dashboard.py`, `run_gradio_dashboard.py`
- **Requirements**: `dashboard_requirements.txt`, `dash_requirements.txt`, `gradio_requirements.txt`

### Visualizations:
- `brand_distribution.png` - Top brands by number of models
- `range_vs_battery.png` - Range vs battery capacity scatter plot
- `speed_vs_range.png` - Speed vs range scatter plot

### Dependencies:
- `requirements.txt` - Python package requirements for notebook

## 🔍 Key Insights Discovered

### Market Overview:
- **478 EV models** from **59 different brands**
- **Average range**: 393 km
- **Average battery capacity**: 74.0 kWh
- **Average top speed**: 185 km/h

### Top Performers:
- **Longest Range**: Mercedes-Benz EQS 450+ (685 km)
- **Fastest Speed**: Maserati GranTurismo Folgore (325 km/h)
- **Quickest Acceleration**: Porsche Taycan Turbo GT Weissach (2.2s)
- **Largest Battery**: Mercedes-Benz EQS variants (118.0 kWh)

### Market Leaders:
- **Mercedes-Benz**: 42 models (most diverse lineup)
- **Audi**: 28 models
- **Porsche**: 26 models
- **Volkswagen**: 23 models
- **Ford**: 22 models

### Technology Trends:
- **99.6%** of vehicles use CCS charging
- **Average fast charging**: 125 kW
- **Most efficient**: Tesla Model 3 variants (7.36 km/kWh)

## 🛠️ Setup and Usage

### Prerequisites:
```bash
pip install -r requirements.txt
```

### Running the Analysis:

#### Option 1: Interactive Dashboards (Choose Your Style)

**🚀 Streamlit Dashboard (Traditional)**
```bash
pip install -r dashboard_requirements.txt
python run_dashboard.py
# OR
streamlit run ev_dashboard.py
```

**🤖 Dash Dashboard (ML-Powered)**
```bash
pip install -r dash_requirements.txt
python run_dash_dashboard.py
```

**🎯 Gradio Dashboard (AI-Focused)**
```bash
pip install -r gradio_requirements.txt
python run_gradio_dashboard.py
```

#### Option 2: Jupyter Notebook
```bash
jupyter notebook electric_vehicle_analysis_2025.ipynb
```

### For Kaggle:
1. Upload the CSV file as a dataset
2. Create a new notebook
3. Copy the notebook content
4. Update the data loading path
5. Run all cells

## 🚗 Interactive Dashboard Comparison

### 📊 Dashboard Options:

| Feature | Streamlit | Dash | Gradio |
|---------|-----------|------|--------|
| **Style** | Traditional | ML-Powered | AI-Focused |
| **Port** | 8501 | 8050 | 7860 |
| **Learning Curve** | Easy | Medium | Easy |
| **Customization** | High | Very High | Medium |
| **ML Integration** | Basic | Advanced | Advanced |
| **UI/UX** | Clean | Professional | Modern |

### 🚀 Streamlit Dashboard Features:
- **8 Analysis Pages**: Overview, Brand Analysis, Performance, Battery & Range, Charging, Segments, Trends, Data Explorer
- **Interactive Features**: Dynamic filtering, real-time updates, data export
- **Charts**: Scatter plots, bar charts, histograms, correlation matrices
- **Best For**: Traditional data analysis, presentations, business users

### 🤖 Dash Dashboard Features:
- **Advanced ML**: K-means clustering, predictive analytics, correlation analysis
- **Interactive 3D**: 3D scatter plots, surface plots, radar charts
- **Real-time Predictions**: Range prediction model, performance forecasting
- **Professional Layout**: Tabbed interface, advanced styling, responsive design
- **Best For**: Data scientists, researchers, advanced analytics

### 🎯 Gradio Dashboard Features:
- **AI Insights**: Natural language queries, AI-generated insights, smart recommendations
- **Performance Predictor**: ML-powered range prediction, brand analysis
- **Smart EV Finder**: AI recommendation system, optimal EV selection
- **Modern Interface**: Clean UI, intuitive controls, mobile-friendly
- **Best For**: AI enthusiasts, modern interfaces, quick insights

## 📈 Analysis Sections

### 1. Data Exploration & Cleaning
- Dataset overview and structure
- Data type conversion and cleaning
- Missing value handling

### 2. Brand Analysis
- Market share by manufacturer
- Performance comparison across brands
- Brand positioning analysis

### 3. Performance Analysis
- Speed vs range correlations
- Acceleration analysis
- Performance by segment

### 4. Battery Technology
- Capacity distribution
- Efficiency metrics
- Charging infrastructure

### 5. Market Segments
- Vehicle category analysis
- Body type distribution
- Segment performance comparison

### 6. Interactive Visualizations
- Plotly charts for exploration
- 3D scatter plots
- Interactive dashboards

## 🎯 Key Questions Answered

1. **What are the performance trends across different vehicle segments?**
2. **How do battery capacities and ranges vary by manufacturer?**
3. **Which brands lead in efficiency and charging capabilities?**
4. **What are the market trends in vehicle sizes and body types?**
5. **How do traditional manufacturers compare to emerging Chinese brands?**

## 📊 Visualization Examples

The analysis includes:
- **Bar charts** for brand comparisons
- **Scatter plots** for performance correlations
- **Histograms** for distribution analysis
- **Box plots** for segment comparisons
- **Interactive plots** for exploration

## 🔧 Customization

The notebook is designed to be easily customizable:
- Modify analysis parameters
- Add new visualizations
- Include additional metrics
- Extend to other datasets

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Feel free to contribute by:
- Adding new analysis sections
- Improving visualizations
- Enhancing the documentation
- Reporting issues or bugs

## 📞 Contact

For questions or suggestions about this analysis, please open an issue in the repository.

---

**Note**: This analysis is based on 2025 model year data and represents the current state of the electric vehicle market. Results may vary as new models are released and specifications are updated.