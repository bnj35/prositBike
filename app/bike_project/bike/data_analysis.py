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

def analyze_launch_strategy(df: pd.DataFrame) -> dict:
    """Analyze data to determine the best launch strategy for bike rental business."""
    launch_analysis = {}
    
    if 'cnt' in df.columns:
        # Find optimal conditions for high demand
        high_demand_days = df[df['cnt'] >= df['cnt'].quantile(0.8)]  # Top 20% demand days
        
        # Season analysis for launch timing
        if 'season' in df.columns:
            season_performance = df.groupby('season')['cnt'].agg(['mean', 'std', 'count']).round(2)
            season_performance['stability_score'] = (season_performance['mean'] / season_performance['std']).round(2)
            launch_analysis['season_recommendation'] = {
                'data': season_performance.to_dict('index'),
                'best_season': season_performance['stability_score'].idxmax(),
                'best_season_avg': season_performance.loc[season_performance['stability_score'].idxmax(), 'mean']
            }
        
        # Weather conditions for launch
        if 'weathersit' in df.columns:
            weather_analysis = df.groupby('weathersit')['cnt'].agg(['mean', 'count', 'std']).round(2)
            weather_analysis['reliability'] = (weather_analysis['mean'] / weather_analysis['std']).round(2)
            launch_analysis['weather_recommendation'] = {
                'data': weather_analysis.to_dict('index'),
                'best_weather': weather_analysis['reliability'].idxmax()
            }
        
        # Day of week analysis
        if 'weekday' in df.columns:
            weekday_stats = df.groupby('weekday')['cnt'].agg(['mean', 'std', 'count']).round(2)
            weekday_stats['consistency'] = (weekday_stats['mean'] / weekday_stats['std']).round(2)
            
            # Map weekday numbers to names
            weekday_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 
                           4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
            
            launch_analysis['weekday_recommendation'] = {
                'data': weekday_stats.to_dict('index'),
                'best_weekday': weekday_stats['consistency'].idxmax(),
                'best_weekday_name': weekday_names.get(weekday_stats['consistency'].idxmax(), 'Unknown'),
                'worst_weekday_name': weekday_names.get(weekday_stats['consistency'].idxmin(), 'Unknown')
            }
        
        # Temperature sweet spot
        if 'temp' in df.columns:
            # Find temperature range with highest average rentals
            df['temp_bins'] = pd.cut(df['temp'], bins=5, labels=['Very Cold', 'Cold', 'Mild', 'Warm', 'Hot'])
            temp_analysis = df.groupby('temp_bins')['cnt'].agg(['mean', 'count']).round(2)
            launch_analysis['temperature_recommendation'] = {
                'data': temp_analysis.to_dict('index'),
                'optimal_temp_range': temp_analysis['mean'].idxmax()
            }
        
        # Working day vs non-working day
        if 'workingday' in df.columns:
            workday_stats = df.groupby('workingday')['cnt'].agg(['mean', 'std', 'count']).round(2)
            launch_analysis['workingday_insight'] = {
                'working_day_avg': workday_stats.loc[1, 'mean'] if 1 in workday_stats.index else 0,
                'non_working_day_avg': workday_stats.loc[0, 'mean'] if 0 in workday_stats.index else 0,
                'recommendation': 'working_day' if (1 in workday_stats.index and workday_stats.loc[1, 'mean'] > workday_stats.loc[0, 'mean']) else 'weekend'
            }
        
        # Market penetration analysis (casual vs registered)
        if 'casual' in df.columns and 'registered' in df.columns:
            casual_growth_potential = df['casual'].std() / df['casual'].mean()  # Higher variation = more growth potential
            registered_stability = df['registered'].mean() / df['registered'].std()  # Lower variation = more stable
            
            launch_analysis['market_penetration'] = {
                'casual_growth_potential': round(casual_growth_potential, 3),
                'registered_stability': round(registered_stability, 3),
                'strategy': 'focus_on_casual' if casual_growth_potential > 0.5 else 'focus_on_registered'
            }
    
    return launch_analysis

def generate_launch_recommendation(df: pd.DataFrame) -> dict:
    """Generate comprehensive launch recommendation based on all analysis."""
    recommendation = {}
    
    # Get all analysis data
    rental_stats = analyze_rentals(df)
    weather_stats = analyze_weather_impact(df)
    seasonal_stats = analyze_seasonal_patterns(df)
    launch_stats = analyze_launch_strategy(df)
    
    # Compile recommendation
    recommendation['executive_summary'] = {
        'total_market_size': rental_stats.get('total_rentals', 0),
        'daily_average_demand': round(rental_stats.get('avg_daily_rentals', 0), 0),
        'peak_demand': rental_stats.get('max_daily_rentals', 0)
    }
    
    # Best launch conditions
    best_conditions = {}
    
    if 'season_recommendation' in launch_stats:
        best_season = launch_stats['season_recommendation']['best_season']
        season_names = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
        best_conditions['season'] = season_names.get(best_season, f'Season {best_season}')
        best_conditions['season_avg_demand'] = launch_stats['season_recommendation']['best_season_avg']
    
    if 'weekday_recommendation' in launch_stats:
        best_conditions['weekday'] = launch_stats['weekday_recommendation']['best_weekday_name']
    
    if 'weather_recommendation' in launch_stats:
        weather_desc = {1: 'Clear/Partly Cloudy', 2: 'Misty/Cloudy', 3: 'Light Snow/Rain'}
        best_weather = launch_stats['weather_recommendation']['best_weather']
        best_conditions['weather'] = weather_desc.get(best_weather, f'Weather Type {best_weather}')
    
    if 'temperature_recommendation' in launch_stats:
        best_conditions['temperature'] = launch_stats['temperature_recommendation']['optimal_temp_range']
    
    recommendation['optimal_launch_conditions'] = best_conditions
    
    # Strategic insights
    insights = []
    
    # Temperature correlation insight
    if 'temperature_correlation' in weather_stats:
        temp_corr = weather_stats['temperature_correlation']
        if temp_corr > 0.5:
            insights.append(f"Strong positive correlation with temperature ({temp_corr:.3f}) - launch in warmer months")
        elif temp_corr < -0.3:
            insights.append(f"Negative correlation with temperature ({temp_corr:.3f}) - consider winter launch")
    
    # Market strategy insight
    if 'market_penetration' in launch_stats:
        strategy = launch_stats['market_penetration']['strategy']
        if strategy == 'focus_on_casual':
            insights.append("High casual user growth potential - focus marketing on tourists/occasional riders")
        else:
            insights.append("Stable registered user base - focus on subscription models and regular commuters")
    
    # Working day insight
    if 'workingday_insight' in launch_stats:
        if launch_stats['workingday_insight']['recommendation'] == 'working_day':
            insights.append("Higher demand on working days - target commuters and business users")
        else:
            insights.append("Higher demand on weekends - target leisure and recreational users")
    
    recommendation['strategic_insights'] = insights
    
    return recommendation

def main():
    """Main CLI entry point for bike sharing data analysis."""
    parser = argparse.ArgumentParser(description="Analyze bike sharing CSV data.")
    parser.add_argument("csv", help="Path to CSV file")
    parser.add_argument("--query", help="pandas query expression")
    parser.add_argument("--summary", action="store_true", help="Print summary stats")
    parser.add_argument("--rentals", action="store_true", help="Analyze rental patterns")
    parser.add_argument("--weather", action="store_true", help="Analyze weather impact")
    parser.add_argument("--seasonal", action="store_true", help="Analyze seasonal patterns")
    parser.add_argument("--launch", action="store_true", help="Analyze best launch strategy")
    parser.add_argument("--recommendation", action="store_true", help="Generate comprehensive launch recommendation")
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
        
        if args.launch or args.all_analysis:
            print("\n🚀 LAUNCH STRATEGY ANALYSIS:")
            launch_stats = analyze_launch_strategy(df)
            for key, value in launch_stats.items():
                print(f"\n{key.replace('_', ' ').title()}:")
                if isinstance(value, dict):
                    for subkey, subval in value.items():
                        if isinstance(subval, dict):
                            print(f"  {subkey}:")
                            for subsubkey, subsubval in subval.items():
                                print(f"    {subsubkey}: {subsubval}")
                        else:
                            print(f"  {subkey}: {subval}")
                else:
                    print(f"  {value}")
            print("-" * 50)
        
        if args.recommendation or args.all_analysis:
            print("\n💡 LAUNCH RECOMMENDATION:")
            recommendation = generate_launch_recommendation(df)
            
            print("\n📈 Executive Summary:")
            for key, value in recommendation['executive_summary'].items():
                print(f"  {key.replace('_', ' ').title()}: {value:,}" if isinstance(value, (int, float)) else f"  {key.replace('_', ' ').title()}: {value}")
            
            print("\n🎯 Optimal Launch Conditions:")
            for key, value in recommendation['optimal_launch_conditions'].items():
                print(f"  Best {key.title()}: {value}")
            
            print("\n🧠 Strategic Insights:")
            for i, insight in enumerate(recommendation['strategic_insights'], 1):
                print(f"  {i}. {insight}")
            
            print("-" * 50)
        
        if not any([args.summary, args.rentals, args.weather, args.seasonal, args.launch, args.recommendation, args.all_analysis]):
            print(f"\n📋 FIRST {args.head} ROWS:")
            print(df.head(args.head))
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())