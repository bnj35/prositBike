# Bike Sharing Data Analysis

A Python package for analyzing bike sharing CSV data with weather, seasonal, and rental pattern insights.

## Usage

### Command Line Interface

#### Basic Usage
```bash
# Since it's a dockerize environment don't forget "docker exec" before each in your terminal
# Show first 5 rows
bike-analyze data/day.csv

# Show first 10 rows
bike-analyze data/day.csv --head 10

# Filter data with pandas query
bike-analyze data/day.csv --query "cnt > 5000"
```

#### Analysis Options
```bash
#  Since it's a dockerize environment don't forget "docker exec" before each in your terminal
# Summary statistics
bike-analyze data/day.csv --summary

# Rental pattern analysis
bike-analyze data/day.csv --rentals

# Weather impact analysis
bike-analyze data/day.csv --weather

# Analyze seasonal patterns
bike-analyze data/day.csv --seasonal

# Launch strategy analysis 
bike-analyze data/day.csv --launch

# Comprehensive launch recommendation 
bike-analyze data/day.csv --recommendation

# Run all analyses
bike-analyze data/day.csv --all-analysis

# Predict best historical launch day
docker exec bike-analysis python -m bike.data_analysis /data/day.csv --predict

# Get comprehensive analysis report
docker exec bike-analysis python -m bike.data_analysis /data/day.csv --prediction-report

# Predict future launch dates
docker exec bike-analysis python -m bike.data_analysis /data/day.csv --future-dates --start-date "2025-06-01"

# Run all predictions together
docker exec bike-analysis python -m bike.data_analysis /data/day.csv --predict --prediction-report --future-dates
```

#### Plotting / Graphs
```bash
# Generate histograms (defaults: cnt, temp, hum, windspeed) and save to the container's /data/plots
docker exec bike-analysis python -m bike.graphs /data/day.csv --hist --out /data/plots

# Generate a regression plot (default x=temp, y=cnt)
docker exec bike-analysis python -m bike.graphs /data/day.csv --reg --x temp --y cnt --out /data/plots

# Generate both histograms and regression together
docker exec bike-analysis python -m bike.graphs /data/day.csv --hist --reg --x temp --y cnt --out /data/plots

# Specify custom histogram columns
docker exec bike-analysis python -m bike.graphs /data/day.csv --hist --hist-cols cnt hum --out /data/plots
 
# Generate presentation-ready plots (time series, monthly, weekday, holiday, boxplots, heatmap)
docker exec bike-analysis python -m bike.graphs /data/day.csv --presentation --out /data/plots
```

#### Combined Usage
```bash
# Filter high-rental days and analyze patterns
bike-analyze data/day.csv --query "cnt > 4000" --all-analysis
```

### Docker Usage
```bash
# Build and start the analysis environment
docker compose up --build

# Access Jupyter Lab at http://localhost:8888
```

### Development
```bash
# Install in development mode
pip install -e .[dev]

# Run tests
pytest tests/
```

## Data Description

The bike sharing dataset contains the following key columns:
- **cnt**: Total bike rentals (casual + registered)
- **casual**: Casual user rentals
- **registered**: Registered user rentals
- **season**: Season (1:spring, 2:summer, 3:fall, 4:winter)
- **weathersit**: Weather situation (1:clear, 2:mist, 3:light snow/rain)
- **temp**: Normalized temperature
- **hum**: Normalized humidity
- **windspeed**: Normalized wind speed
- **weekday**: Day of week
- **workingday**: Working day indicator
- **holiday**: Holiday indicator

## Analysis Features

1. **Rental Analysis**: Total rentals, daily averages, casual vs registered users
2. **Weather Impact**: Correlation with temperature, humidity, wind speed
3. **Seasonal Patterns**: Analysis by season, weekday, working days, holidays
4. **Launch Strategy**: Optimal launch timing, market conditions, stability analysis
5. **Launch Recommendations**: Comprehensive business launch strategy based on data
6. **Custom Filtering**: Use pandas queries to filter data before analysis

## Project Structure

```
prositBike/
├── docker-compose.yml       # Docker Compose configuration
├── Dockerfile              # Docker image definition
├── README.md               # This file
├── data/                   # CSV data files
│   ├── day.csv            # Daily bike sharing data
│   ├── hour.csv           # Hourly bike sharing data
│   └── Readme.txt         # Data description
└── app/
    └── bike_project/       # Python package
        ├── pyproject.toml  # Package configuration
        ├── bike/           # Main package
        │   ├── __init__.py
        │   └── data_analysis.py  # Analysis functions
        └── tests/          # Unit tests
            └── test_data_analysis.py
```