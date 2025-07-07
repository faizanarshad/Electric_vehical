# Electric Vehicle Analysis Project Structure

## 📁 Directory Organization

```
Electric_vehical/
├── 📊 data/                          # Data files
│   ├── raw/                          # Original data files
│   │   └── electric_vehicles_spec_2025.csv.csv
│   ├── processed/                    # Processed and cleaned data
│   │   ├── electric_vehicles_analyzed_2025.csv
│   │   └── ev_summary_statistics_2025.csv
│   └── cleaned/                      # Final cleaned datasets
├── 📓 notebooks/                     # Jupyter notebooks
│   └── electric_vehicle_analysis_2025.ipynb
├── 🔧 src/                          # Source code
│   ├── analysis/                     # Data analysis modules
│   ├── visualization/                # Visualization modules
│   ├── dashboards/                   # Dashboard applications
│   │   ├── ev_streamlit_dashboard.py
│   │   ├── ev_dash_dashboard.py
│   │   ├── ev_gradio_dashboard.py
│   │   └── ev_dashboard.py
│   └── utils/                        # Utility functions
├── 🚀 scripts/                       # Launcher scripts
│   ├── run_dashboard.py
│   ├── run_dash_dashboard.py
│   └── run_gradio_dashboard.py
├── ⚙️ config/                        # Configuration files
│   ├── requirements.txt
│   ├── dashboard_requirements.txt
│   ├── dash_requirements.txt
│   └── gradio_requirements.txt
├── 📈 results/                       # Output files
│   ├── figures/                      # Generated plots and charts
│   │   ├── brand_distribution.png
│   │   ├── range_vs_battery.png
│   │   └── speed_vs_range.png
│   └── models/                       # Trained models
├── 📚 docs/                          # Documentation
│   ├── api/                          # API documentation
│   └── user_guide/                   # User guides
├── 🧪 tests/                         # Test files
├── 🔧 setup.py                       # Project setup script
├── 🚀 run_project.py                 # Main launcher
├── 📋 requirements.txt               # Main dependencies
├── 📖 README.md                      # Project documentation
└── 📋 PROJECT_STRUCTURE.md           # This file
```

## 🎯 Purpose of Each Directory

### 📊 Data Directory
- **raw/**: Contains original, unmodified data files
- **processed/**: Contains data that has been cleaned and processed
- **cleaned/**: Contains final, analysis-ready datasets

### 📓 Notebooks Directory
- Jupyter notebooks for exploratory data analysis
- Data cleaning and preprocessing workflows
- Model development and testing

### 🔧 Source Code (src/)
- **analysis/**: Statistical analysis and data processing modules
- **visualization/**: Chart and plot generation functions
- **dashboards/**: Interactive web applications
- **utils/**: Helper functions and utilities

### 🚀 Scripts Directory
- Launcher scripts for different dashboards
- Automation scripts for data processing
- Deployment scripts

### ⚙️ Config Directory
- Requirements files for different components
- Configuration files for dashboards
- Environment settings

### 📈 Results Directory
- **figures/**: Generated visualizations and charts
- **models/**: Trained machine learning models
- Analysis outputs and reports

### 📚 Documentation
- **api/**: Technical documentation for developers
- **user_guide/**: User manuals and tutorials

## 🚀 How to Run the Project

1. **Setup**: `python setup.py`
2. **Run Main Launcher**: `python run_project.py`
3. **Run Individual Dashboards**:
   - Streamlit: `python scripts/run_dashboard.py`
   - Dash: `python scripts/run_dash_dashboard.py`
   - Gradio: `python scripts/run_gradio_dashboard.py`

## 📋 Best Practices

- Keep raw data in `data/raw/` and never modify it
- Process data through the pipeline: raw → processed → cleaned
- Use relative paths in code to maintain portability
- Document all data transformations and model decisions
- Version control all code but exclude large data files and models 