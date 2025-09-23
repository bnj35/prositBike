from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def load_csv(path: str | Path) -> pd.DataFrame:
    """Load a bike sharing CSV file into a pandas DataFrame."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    
    df = pd.read_csv(p)
    
    # Convert date column if it exists
    if 'dteday' in df.columns:
        df['dteday'] = pd.to_datetime(df['dteday'])
    
    return df

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Generate summary statistics for bike sharing data."""
    return df.describe(include="all").transpose()

def analyze_rentals(df: pd.DataFrame) -> dict:
    """Analyze bike rental patterns."""
    analysis = {}
    
    if 'cnt' in df.columns:
        analysis['total_rentals'] = df['cnt'].sum()
        analysis['avg_daily_rentals'] = df['cnt'].mean()
        analysis['max_daily_rentals'] = df['cnt'].max()
        analysis['min_daily_rentals'] = df['cnt'].min()
    
    if 'casual' in df.columns and 'registered' in df.columns:
        analysis['casual_vs_registered'] = {
            'casual_total': df['casual'].sum(),
            'registered_total': df['registered'].sum(),
            'casual_percentage': (df['casual'].sum() / (df['casual'].sum() + df['registered'].sum())) * 100
        }
    
    return analysis

def analyze_weather_impact(df: pd.DataFrame) -> dict:
    """Analyze impact of weather on bike rentals."""
    weather_analysis = {}
    
    if 'cnt' in df.columns and 'weathersit' in df.columns:
        weather_rentals = df.groupby('weathersit')['cnt'].agg(['mean', 'count', 'sum'])
        weather_analysis['by_weather_situation'] = weather_rentals.to_dict('index')
    
    if 'cnt' in df.columns and 'temp' in df.columns:
        # Temperature correlation
        temp_corr = df['cnt'].corr(df['temp'])
        weather_analysis['temperature_correlation'] = temp_corr
    
    if 'cnt' in df.columns and 'hum' in df.columns:
        # Humidity correlation
        hum_corr = df['cnt'].corr(df['hum'])
        weather_analysis['humidity_correlation'] = hum_corr
    
    if 'cnt' in df.columns and 'windspeed' in df.columns:
        # Wind speed correlation
        wind_corr = df['cnt'].corr(df['windspeed'])
        weather_analysis['windspeed_correlation'] = wind_corr
    
    return weather_analysis

def analyze_seasonal_patterns(df: pd.DataFrame) -> dict:
    """Analyze seasonal and temporal patterns."""
    seasonal_analysis = {}
    
    if 'cnt' in df.columns and 'season' in df.columns:
        seasonal_rentals = df.groupby('season')['cnt'].agg(['mean', 'sum', 'count'])
        seasonal_analysis['by_season'] = seasonal_rentals.to_dict('index')
    
    if 'cnt' in df.columns and 'weekday' in df.columns:
        weekday_rentals = df.groupby('weekday')['cnt'].agg(['mean', 'sum'])
        seasonal_analysis['by_weekday'] = weekday_rentals.to_dict('index')
    
    if 'cnt' in df.columns and 'workingday' in df.columns:
        workday_analysis = df.groupby('workingday')['cnt'].agg(['mean', 'sum'])
        seasonal_analysis['working_vs_non_working'] = workday_analysis.to_dict('index')
    
    if 'cnt' in df.columns and 'holiday' in df.columns:
        holiday_analysis = df.groupby('holiday')['cnt'].agg(['mean', 'sum'])
        seasonal_analysis['holiday_vs_normal'] = holiday_analysis.to_dict('index')
    
    return seasonal_analysis

def filter_rows(df: pd.DataFrame, query_expr: str) -> pd.DataFrame:
    """Filter DataFrame rows using a pandas query expression."""
    return df.query(query_expr)

def main():
    """Main CLI entry point for bike sharing data analysis."""
    parser = argparse.ArgumentParser(description="Analyze bike sharing CSV data.")
    parser.add_argument("csv", help="Path to CSV file")
    parser.add_argument("--query", help="pandas query expression")
    parser.add_argument("--summary", action="store_true", help="Print summary stats")
    parser.add_argument("--rentals", action="store_true", help="Analyze rental patterns")
    parser.add_argument("--weather", action="store_true", help="Analyze weather impact")
    parser.add_argument("--seasonal", action="store_true", help="Analyze seasonal patterns")
    parser.add_argument("--all-analysis", action="store_true", help="Run all analysis types")
    parser.add_argument("--head", type=int, default=5, help="Show first N rows (ignored with analysis flags)")
    args = parser.parse_args()

    try:
        df = load_csv(args.csv)
        print(f"Loaded {len(df)} records from {args.csv}")
        print(f"Columns: {', '.join(df.columns)}")
        print("-" * 50)
        
        if args.query:
            df = filter_rows(df, args.query)
            print(f"Filtered to {len(df)} records")
            print("-" * 50)

        if args.summary or args.all_analysis:
            print("\n📊 SUMMARY STATISTICS:")
            print(summarize(df))
            print("-" * 50)
        
        if args.rentals or args.all_analysis:
            print("\n🚴 RENTAL ANALYSIS:")
            rental_stats = analyze_rentals(df)
            for key, value in rental_stats.items():
                if isinstance(value, dict):
                    print(f"{key.replace('_', ' ').title()}:")
                    for subkey, subval in value.items():
                        print(f"  {subkey}: {subval:.2f}" if isinstance(subval, (int, float)) else f"  {subkey}: {subval}")
                else:
                    print(f"{key.replace('_', ' ').title()}: {value:.2f}" if isinstance(value, (int, float)) else f"{key.replace('_', ' ').title()}: {value}")
            print("-" * 50)
        
        if args.weather or args.all_analysis:
            print("\n🌤️ WEATHER IMPACT ANALYSIS:")
            weather_stats = analyze_weather_impact(df)
            for key, value in weather_stats.items():
                if isinstance(value, dict):
                    print(f"{key.replace('_', ' ').title()}:")
                    for subkey, subval in value.items():
                        print(f"  Weather {subkey}: {subval}")
                else:
                    print(f"{key.replace('_', ' ').title()}: {value:.4f}")
            print("-" * 50)
        
        if args.seasonal or args.all_analysis:
            print("\n📅 SEASONAL PATTERN ANALYSIS:")
            seasonal_stats = analyze_seasonal_patterns(df)
            for key, value in seasonal_stats.items():
                print(f"{key.replace('_', ' ').title()}:")
                for subkey, subval in value.items():
                    print(f"  {subkey}: {subval}")
            print("-" * 50)
        
        if not any([args.summary, args.rentals, args.weather, args.seasonal, args.all_analysis]):
            print(f"\n📋 FIRST {args.head} ROWS:")
            print(df.head(args.head))
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())