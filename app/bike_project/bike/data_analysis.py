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
    
    # Monthly analysis - very important for launch timing
    if 'cnt' in df.columns and 'mnth' in df.columns:
        monthly_stats = df.groupby('mnth')['cnt'].agg(['mean', 'std', 'count', 'min', 'max']).round(2)
        monthly_stats['stability'] = (monthly_stats['mean'] / monthly_stats['std']).round(2)
        
        # Add month names
        month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
                      7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
        
        monthly_analysis = {}
        for month, stats in monthly_stats.iterrows():
            monthly_analysis[month] = {
                'month_name': month_names.get(month, f'Month {month}'),
                'avg_rentals': float(stats['mean']),
                'stability_score': float(stats['stability']),
                'min_rentals': float(stats['min']),
                'max_rentals': float(stats['max']),
                'days_analyzed': int(stats['count'])
            }
        
        seasonal_analysis['by_month'] = monthly_analysis
        seasonal_analysis['best_month'] = int(monthly_stats['stability'].idxmax())
        seasonal_analysis['best_month_name'] = month_names.get(int(monthly_stats['stability'].idxmax()), 'Unknown')
    
    if 'cnt' in df.columns and 'weekday' in df.columns:
        weekday_rentals = df.groupby('weekday')['cnt'].agg(['mean', 'sum'])
        seasonal_analysis['by_weekday'] = weekday_rentals.to_dict('index')
    
    if 'cnt' in df.columns and 'workingday' in df.columns:
        workday_analysis = df.groupby('workingday')['cnt'].agg(['mean', 'sum'])
        seasonal_analysis['working_vs_non_working'] = workday_analysis.to_dict('index')
    
    # Enhanced holiday analysis
    if 'cnt' in df.columns and 'holiday' in df.columns:
        holiday_detailed = df.groupby('holiday')['cnt'].agg(['mean', 'std', 'count', 'min', 'max']).round(2)
        
        holiday_analysis = {}
        for is_holiday, stats in holiday_detailed.iterrows():
            holiday_type = 'Holiday' if is_holiday == 1 else 'Regular Day'
            holiday_analysis[is_holiday] = {
                'day_type': holiday_type,
                'avg_rentals': float(stats['mean']),
                'variability': float(stats['std']),
                'min_rentals': float(stats['min']),
                'max_rentals': float(stats['max']),
                'sample_size': int(stats['count'])
            }
        
        seasonal_analysis['holiday_detailed'] = holiday_analysis
        
        # Holiday impact calculation
        if 0 in holiday_detailed.index and 1 in holiday_detailed.index:
            holiday_impact = ((holiday_detailed.loc[1, 'mean'] - holiday_detailed.loc[0, 'mean']) / holiday_detailed.loc[0, 'mean'] * 100)
            seasonal_analysis['holiday_impact_percent'] = round(holiday_impact, 2)
            seasonal_analysis['holiday_recommendation'] = 'avoid_holidays' if holiday_impact < -10 else 'neutral' if abs(holiday_impact) < 10 else 'prefer_holidays'
    
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
        
        # Monthly launch timing analysis
        if 'mnth' in df.columns:
            monthly_launch = df.groupby('mnth')['cnt'].agg(['mean', 'std', 'count']).round(2)
            monthly_launch['launch_score'] = (monthly_launch['mean'] / monthly_launch['std']).round(2)
            
            month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
                          7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
            
            launch_analysis['monthly_recommendation'] = {
                'best_month_number': int(monthly_launch['launch_score'].idxmax()),
                'best_month_name': month_names.get(int(monthly_launch['launch_score'].idxmax()), 'Unknown'),
                'best_month_avg': monthly_launch.loc[monthly_launch['launch_score'].idxmax(), 'mean'],
                'best_month_stability': monthly_launch.loc[monthly_launch['launch_score'].idxmax(), 'launch_score'],
                'worst_month_name': month_names.get(int(monthly_launch['launch_score'].idxmin()), 'Unknown'),
                'monthly_data': {month_names.get(month, f'Month {month}'): {
                    'avg_rentals': float(stats['mean']),
                    'stability': float(stats['launch_score'])
                } for month, stats in monthly_launch.iterrows()}
            }
        
        # Holiday impact on launch
        if 'holiday' in df.columns:
            holiday_impact = df.groupby('holiday')['cnt'].agg(['mean', 'std', 'count']).round(2)
            
            if 0 in holiday_impact.index and 1 in holiday_impact.index:
                holiday_effect = ((holiday_impact.loc[1, 'mean'] - holiday_impact.loc[0, 'mean']) / holiday_impact.loc[0, 'mean'] * 100)
                
                launch_analysis['holiday_impact'] = {
                    'regular_day_avg': holiday_impact.loc[0, 'mean'],
                    'holiday_avg': holiday_impact.loc[1, 'mean'],
                    'impact_percentage': round(holiday_effect, 2),
                    'recommendation': 'avoid_holidays' if holiday_effect < -15 else 'neutral' if abs(holiday_effect) < 10 else 'prefer_holidays',
                    'reasoning': _get_holiday_reasoning(holiday_effect)
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

def _get_holiday_reasoning(holiday_effect):
    """Get reasoning for holiday recommendation."""
    if holiday_effect < -15:
        return "Holidays show significantly lower demand - avoid launching on holidays"
    elif holiday_effect > 15:
        return "Holidays show significantly higher demand - consider holiday launch"
    else:
        return "Holiday impact is minimal - timing flexibility"

def predict_optimal_launch_day(df: pd.DataFrame) -> dict:
    """Predict the best specific day to launch based on historical patterns."""
    prediction = {}
    
    if 'dteday' in df.columns and 'cnt' in df.columns:
        # Add derived features for prediction
        df_pred = df.copy()
        df_pred['month'] = df_pred['dteday'].dt.month
        df_pred['day'] = df_pred['dteday'].dt.day
        df_pred['year'] = df_pred['dteday'].dt.year
        
        # Create composite score for each day based on multiple factors
        df_pred['launch_score'] = 0
        
        # Factor 1: Base rental demand (30% weight)
        if 'cnt' in df_pred.columns:
            rental_normalized = (df_pred['cnt'] - df_pred['cnt'].min()) / (df_pred['cnt'].max() - df_pred['cnt'].min())
            df_pred['launch_score'] += rental_normalized * 0.3
        
        # Factor 2: Weather conditions (25% weight)
        if 'weathersit' in df_pred.columns:
            weather_score = df_pred['weathersit'].map({1: 1.0, 2: 0.6, 3: 0.2})  # Clear=1, Misty=0.6, Rain=0.2
            df_pred['launch_score'] += weather_score * 0.25
        
        # Factor 3: Temperature (20% weight)
        if 'temp' in df_pred.columns:
            df_pred['launch_score'] += df_pred['temp'] * 0.2
        
        # Factor 4: Non-holiday bonus (15% weight)
        if 'holiday' in df_pred.columns:
            holiday_penalty = df_pred['holiday'].map({0: 1.0, 1: 0.0})  # Non-holiday=1, Holiday=0
            df_pred['launch_score'] += holiday_penalty * 0.15
        
        # Factor 5: Weekend bonus (10% weight)
        if 'weekday' in df_pred.columns:
            weekend_bonus = df_pred['weekday'].map({5: 1.0, 6: 0.9, 4: 0.8, 3: 0.7, 2: 0.6, 1: 0.5, 0: 0.4})
            df_pred['launch_score'] += weekend_bonus * 0.1
        
        # Find top 10 launch days
        top_days = df_pred.nlargest(10, 'launch_score')[
            ['dteday', 'cnt', 'launch_score', 'weathersit', 'temp', 'weekday', 'holiday', 'season', 'mnth']
        ].copy()
        
        # Add descriptive information
        season_names = {1: 'Spring', 2: 'Summer', 3: 'Fall', 4: 'Winter'}
        weekday_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
        weather_names = {1: 'Clear', 2: 'Misty', 3: 'Rain/Snow'}
        month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 
                      7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
        
        prediction['top_launch_days'] = []
        for idx, row in top_days.iterrows():
            day_info = {
                'date': row['dteday'].strftime('%Y-%m-%d'),
                'day_of_week': weekday_names.get(row['weekday'], 'Unknown'),
                'month': month_names.get(row['mnth'], 'Unknown'),
                'season': season_names.get(row['season'], 'Unknown'),
                'actual_rentals': int(row['cnt']),
                'launch_score': round(row['launch_score'], 3),
                'weather': weather_names.get(row['weathersit'], 'Unknown'),
                'temperature': round(row['temp'], 3),
                'is_holiday': bool(row['holiday']),
                'rank': len(prediction['top_launch_days']) + 1
            }
            prediction['top_launch_days'].append(day_info)
        
        # Best overall day
        best_day = top_days.iloc[0]
        prediction['recommended_launch_day'] = {
            'date': best_day['dteday'].strftime('%Y-%m-%d'),
            'day_of_week': weekday_names.get(best_day['weekday'], 'Unknown'),
            'month': month_names.get(best_day['mnth'], 'Unknown'),
            'season': season_names.get(best_day['season'], 'Unknown'),
            'expected_rentals': int(best_day['cnt']),
            'launch_score': round(best_day['launch_score'], 3),
            'weather_conditions': weather_names.get(best_day['weathersit'], 'Unknown'),
            'temperature': round(best_day['temp'], 3),
            'is_holiday': bool(best_day['holiday'])
        }
        
        # Future prediction based on patterns
        prediction['future_prediction_model'] = create_future_launch_prediction(df_pred)
    
    return prediction

def create_future_launch_prediction(df: pd.DataFrame) -> dict:
    """Create a model to predict future optimal launch dates."""
    future_model = {}
    
    # Analyze patterns by month and weekday
    monthly_patterns = df.groupby(['mnth', 'weekday'])['launch_score'].mean().reset_index()
    monthly_patterns = monthly_patterns.sort_values('launch_score', ascending=False)
    
    # Best month-weekday combinations
    top_combinations = monthly_patterns.head(5)
    
    month_names = {1: 'January', 2: 'February', 3: 'March', 4: 'April', 5: 'May', 6: 'June',
                   7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
    weekday_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}
    
    future_model['best_month_weekday_combinations'] = []
    for _, row in top_combinations.iterrows():
        combo = {
            'month': month_names.get(row['mnth'], f"Month {row['mnth']}"),
            'weekday': weekday_names.get(row['weekday'], f"Weekday {row['weekday']}"),
            'predicted_score': round(row['launch_score'], 3)
        }
        future_model['best_month_weekday_combinations'].append(combo)
    
    # Weather-based prediction
    weather_impact = df.groupby('weathersit')['launch_score'].mean()
    future_model['weather_recommendations'] = {
        'clear_weather_score': round(weather_impact.get(1, 0), 3),
        'misty_weather_score': round(weather_impact.get(2, 0), 3),
        'poor_weather_score': round(weather_impact.get(3, 0), 3),
        'recommendation': 'Launch only on clear weather days for maximum success'
    }
    
    # Temperature sweet spot
    temp_bins = pd.cut(df['temp'], bins=5, labels=['Very Cold', 'Cold', 'Mild', 'Warm', 'Hot'])
    temp_scores = df.groupby(temp_bins)['launch_score'].mean()
    future_model['temperature_recommendations'] = {
        temp_range: round(score, 3) for temp_range, score in temp_scores.items()
    }
    
    return future_model

def generate_launch_prediction_report(df: pd.DataFrame) -> dict:
    """Generate a comprehensive launch prediction report."""
    report = {}
    
    # Get prediction analysis
    prediction = predict_optimal_launch_day(df)
    
    # Executive summary
    if 'recommended_launch_day' in prediction:
        best_day = prediction['recommended_launch_day']
        report['executive_summary'] = {
            'recommended_date': best_day['date'],
            'day_of_week': best_day['day_of_week'],
            'month': best_day['month'],
            'expected_performance': f"{best_day['expected_rentals']:,} rentals",
            'confidence_score': f"{best_day['launch_score']:.1%}",
            'weather_forecast': best_day['weather_conditions']
        }
    
    # Top alternatives
    if 'top_launch_days' in prediction:
        report['alternative_dates'] = prediction['top_launch_days'][1:6]  # Top 2-6
    
    # Future guidance
    if 'future_prediction_model' in prediction:
        future_model = prediction['future_prediction_model']
        report['future_launch_guidance'] = {
            'best_combinations': future_model['best_month_weekday_combinations'][:3],
            'weather_strategy': future_model['weather_recommendations']['recommendation'],
            'optimal_temperature': max(future_model['temperature_recommendations'].items(), 
                                     key=lambda x: x[1])[0] if future_model['temperature_recommendations'] else 'Warm'
        }
    
    return report

def generate_launch_recommendation(df: pd.DataFrame) -> dict:
    """Generate launch recommendations based on analysis (legacy function)."""
    # Use the new prediction system
    prediction_report = generate_launch_prediction_report(df)
    
    # Convert to legacy format for compatibility
    recommendation = {
        'summary': prediction_report.get('executive_summary', {}),
        'best_date': prediction_report.get('executive_summary', {}).get('recommended_date', 'Unknown'),
        'alternatives': prediction_report.get('alternative_dates', []),
        'strategy': prediction_report.get('future_launch_guidance', {})
    }
    
    return recommendation
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
    
    if 'monthly_recommendation' in launch_stats:
        best_conditions['month'] = launch_stats['monthly_recommendation']['best_month_name']
        best_conditions['month_avg_demand'] = launch_stats['monthly_recommendation']['best_month_avg']
    
    if 'weather_recommendation' in launch_stats:
        weather_desc = {1: 'Clear/Partly Cloudy', 2: 'Misty/Cloudy', 3: 'Light Snow/Rain'}
        best_weather = launch_stats['weather_recommendation']['best_weather']
        best_conditions['weather'] = weather_desc.get(best_weather, f'Weather Type {best_weather}')
    
    if 'temperature_recommendation' in launch_stats:
        best_conditions['temperature'] = launch_stats['temperature_recommendation']['optimal_temp_range']
    
    if 'holiday_impact' in launch_stats:
        holiday_rec = launch_stats['holiday_impact']['recommendation']
        if holiday_rec == 'avoid_holidays':
            best_conditions['holiday_timing'] = 'Avoid holidays'
        elif holiday_rec == 'prefer_holidays':
            best_conditions['holiday_timing'] = 'Prefer holidays'
        else:
            best_conditions['holiday_timing'] = 'Holiday timing flexible'
    
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
    
    # Monthly timing insight
    if 'monthly_recommendation' in launch_stats:
        best_month = launch_stats['monthly_recommendation']['best_month_name']
        insights.append(f"Optimal launch month is {best_month} based on demand stability and volume")
    
    # Holiday impact insight
    if 'holiday_impact' in launch_stats:
        holiday_reasoning = launch_stats['holiday_impact']['reasoning']
        insights.append(holiday_reasoning)
    
    recommendation['strategic_insights'] = insights
    
    return recommendation

def predict_future_launch_dates(df: pd.DataFrame, start_date='2025-01-01', months_ahead=12) -> dict:
    """Predict optimal future launch dates based on historical patterns."""
    from datetime import datetime, timedelta
    import calendar
    
    future_prediction = {}
    
    # Parse start date
    start = datetime.strptime(start_date, '%Y-%m-%d')
    
    # Analyze historical patterns
    if 'dteday' in df.columns:
        df_analysis = df.copy()
        df_analysis['month'] = df_analysis['dteday'].dt.month
        df_analysis['weekday'] = df_analysis['dteday'].dt.weekday
        df_analysis['season'] = df_analysis['season'].astype(int)
        
        # Calculate monthly patterns
        monthly_performance = df_analysis.groupby('month')['cnt'].agg(['mean', 'std']).reset_index()
        monthly_performance['score'] = (monthly_performance['mean'] / monthly_performance['mean'].max()) * 100
        monthly_performance = monthly_performance.sort_values('score', ascending=False)
        
        # Calculate weekday patterns
        weekday_performance = df_analysis.groupby('weekday')['cnt'].agg(['mean', 'std']).reset_index()
        weekday_performance['score'] = (weekday_performance['mean'] / weekday_performance['mean'].max()) * 100
        
        # Generate future recommendations
        future_dates = []
        current_date = start
        
        for _ in range(months_ahead * 4):  # Check 4 dates per month
            month = current_date.month
            weekday = current_date.weekday()
            
            # Get month score
            month_matches = monthly_performance[monthly_performance['month'] == month]
            month_score = month_matches['score'].iloc[0] if len(month_matches) > 0 else 50
            
            # Get weekday score  
            weekday_matches = weekday_performance[weekday_performance['weekday'] == weekday]
            weekday_score = weekday_matches['score'].iloc[0] if len(weekday_matches) > 0 else 50
            
            # Combine scores
            combined_score = (month_score * 0.6 + weekday_score * 0.4)
            
            # Predict rentals based on historical averages
            month_avg = month_matches['mean'].iloc[0] if len(month_matches) > 0 else 4000
            weekday_avg = weekday_matches['mean'].iloc[0] if len(weekday_matches) > 0 else 4000
            predicted_rentals = int((month_avg + weekday_avg) / 2)
            
            # Season mapping
            season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                         3: 'Spring', 4: 'Spring', 5: 'Spring', 
                         6: 'Summer', 7: 'Summer', 8: 'Summer',
                         9: 'Fall', 10: 'Fall', 11: 'Fall'}
            
            weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            
            future_dates.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'day_of_week': weekday_names[int(weekday)],
                'month': calendar.month_name[int(month)],
                'season': season_map.get(int(month), 'Unknown'),
                'predicted_rentals': predicted_rentals,
                'launch_score': round(combined_score, 1),
                'month_score': round(month_score, 1),
                'weekday_score': round(weekday_score, 1)
            })
            
            # Move to next week
            current_date += timedelta(days=7)
        
        # Sort by launch score and get top recommendations
        future_dates = sorted(future_dates, key=lambda x: x['launch_score'], reverse=True)
        
        future_prediction['top_future_dates'] = future_dates[:10]
        future_prediction['monthly_rankings'] = []
        
        # Monthly breakdown
        monthly_names = {i: calendar.month_name[i] for i in range(1, 13)}
        weekday_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        for _, row in monthly_performance.iterrows():
            future_prediction['monthly_rankings'].append({
                'month': monthly_names[row['month']],
                'average_rentals': int(row['mean']),
                'consistency': round(100 - (row['std'] / row['mean'] * 100), 1),
                'recommendation_score': round(row['score'], 1)
            })
        
        # Best strategies
        best_weekday_idx = int(weekday_performance.sort_values('score', ascending=False).iloc[0]['weekday'])
        future_prediction['launch_strategies'] = {
            'best_month': monthly_names[int(monthly_performance.iloc[0]['month'])],
            'best_weekday': weekday_names[best_weekday_idx],
            'avoid_weather': 'Rain/Snow days',
            'optimal_season': season_map.get(int(monthly_performance.iloc[0]['month']), 'Unknown')
        }
    
    return future_prediction

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
    parser.add_argument("--predict", action="store_true", help="Predict optimal launch day")
    parser.add_argument("--prediction-report", action="store_true", help="Generate comprehensive prediction report")
    parser.add_argument("--future-dates", action="store_true", help="Predict future optimal launch dates")
    parser.add_argument("--start-date", default="2025-01-01", help="Start date for future predictions (YYYY-MM-DD)")
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
                if key == 'best_month':
                    print(f"Best Month Number: {value}")
                elif key == 'best_month_name':
                    print(f"Best Month Name: {value}")
                elif key == 'holiday_impact_percent':
                    print(f"Holiday Impact: {value}%")
                elif key == 'holiday_recommendation':
                    print(f"Holiday Recommendation: {value}")
                else:
                    print(f"{key.replace('_', ' ').title()}:")
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
        
        if args.predict:
            print("\n🔮 OPTIMAL LAUNCH DAY PREDICTION:")
            prediction = predict_optimal_launch_day(df)
            
            if 'recommended_launch_day' in prediction:
                best = prediction['recommended_launch_day']
                print(f"\n🎯 Best Launch Date: {best['date']}")
                print(f"📅 Day of Week: {best['day_of_week']}")
                print(f"🗓️  Month: {best['month']} ({best['season']})")
                print(f"🚴 Expected Rentals: {best['expected_rentals']:,}")
                print(f"⭐ Launch Score: {best['launch_score']:.1%}")
                print(f"🌤️  Weather: {best['weather_conditions']}")
                print(f"🌡️  Temperature: {best['temperature']}")
                print(f"🎉 Holiday: {'Yes' if best['is_holiday'] else 'No'}")
                
                print(f"\n📊 Top 5 Alternative Launch Days:")
                for day in prediction['top_launch_days'][1:6]:
                    print(f"  {day['rank']}. {day['date']} ({day['day_of_week']}) - "
                          f"{day['actual_rentals']:,} rentals, Score: {day['launch_score']:.3f}")
            
            print("-" * 50)
        
        if args.prediction_report:
            print("\n📈 COMPREHENSIVE LAUNCH PREDICTION REPORT:")
            report = generate_launch_prediction_report(df)
            
            if 'executive_summary' in report:
                summary = report['executive_summary']
                print("\n🎯 EXECUTIVE SUMMARY:")
                print(f"  • Recommended Date: {summary.get('recommended_date', 'Unknown')}")
                print(f"  • Day & Month: {summary.get('day_of_week', 'Unknown')}, {summary.get('month', 'Unknown')}")
                print(f"  • Expected Performance: {summary.get('expected_performance', 'Unknown')}")
                print(f"  • Success Probability: {summary.get('confidence_score', 'Unknown')}")
                print(f"  • Weather Conditions: {summary.get('weather_forecast', 'Unknown')}")
            
            if 'alternative_dates' in report:
                print(f"\n📅 TOP ALTERNATIVE DATES:")
                for i, day in enumerate(report['alternative_dates'], 2):
                    print(f"  {i}. {day['date']} ({day['day_of_week']}) - {day['actual_rentals']:,} rentals")
            
            if 'future_launch_guidance' in report:
                guidance = report['future_launch_guidance']
                print(f"\n🔮 FUTURE LAUNCH GUIDANCE:")
                print(f"  • Weather Strategy: {guidance.get('weather_strategy', 'Unknown')}")
                print(f"  • Optimal Temperature: {guidance.get('optimal_temperature', 'Unknown')}")
                
                if 'best_combinations' in guidance:
                    print("  • Best Month-Day Combinations:")
                    for combo in guidance['best_combinations']:
                        print(f"    - {combo['month']} {combo['weekday']} (Score: {combo['predicted_score']:.3f})")
            
            print("-" * 50)
        
        if args.future_dates:
            print("\n🔮 FUTURE LAUNCH DATE PREDICTIONS:")
            future_pred = predict_future_launch_dates(df, args.start_date)
            
            if 'top_future_dates' in future_pred:
                print(f"\n🎯 TOP 10 FUTURE LAUNCH DATES (starting from {args.start_date}):")
                for i, date in enumerate(future_pred['top_future_dates'], 1):
                    print(f"  {i:2d}. {date['date']} ({date['day_of_week']}) - {date['month']} {date['season']}")
                    print(f"      Expected: {date['predicted_rentals']:,} rentals, Score: {date['launch_score']}%")
            
            if 'monthly_rankings' in future_pred:
                print(f"\n📅 MONTHLY PERFORMANCE RANKINGS:")
                for i, month in enumerate(future_pred['monthly_rankings'], 1):
                    print(f"  {i:2d}. {month['month']:10s} - Avg: {month['average_rentals']:,} rentals, "
                          f"Consistency: {month['consistency']}%, Score: {month['recommendation_score']}%")
            
            if 'launch_strategies' in future_pred:
                strategies = future_pred['launch_strategies']
                print(f"\n💡 OPTIMAL LAUNCH STRATEGY:")
                print(f"  • Best Month: {strategies['best_month']}")
                print(f"  • Best Day of Week: {strategies['best_weekday']}")
                print(f"  • Optimal Season: {strategies['optimal_season']}")
                print(f"  • Weather to Avoid: {strategies['avoid_weather']}")
            
            print("-" * 50)
        
        if not any([args.summary, args.rentals, args.weather, args.seasonal, args.launch, args.recommendation, args.predict, args.prediction_report, args.future_dates, args.all_analysis]):
            print(f"\n📋 FIRST {args.head} ROWS:")
            print(df.head(args.head))
            
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())