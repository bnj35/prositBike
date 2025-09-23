import pandas as pd
import pytest
from bike.data_analysis import load_csv, summarize, filter_rows, analyze_rentals, analyze_weather_impact, analyze_seasonal_patterns

def test_load_csv_success(tmp_path):
    """Test successful CSV loading with bike data."""
    csv_file = tmp_path / "test_bikes.csv"
    csv_file.write_text("dteday,season,cnt,casual,registered,temp,weathersit\n2011-01-01,1,985,331,654,0.344,2\n2011-01-02,1,801,131,670,0.363,2\n")
    
    df = load_csv(csv_file)
    assert df.shape == (2, 7)
    assert 'dteday' in df.columns
    assert 'cnt' in df.columns

def test_load_csv_file_not_found():
    """Test loading non-existent CSV file."""
    with pytest.raises(FileNotFoundError):
        load_csv("non_existent_file.csv")

def test_summarize(tmp_path):
    """Test DataFrame summary generation."""
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("cnt,casual,registered\n985,331,654\n801,131,670\n1349,120,1229\n")
    
    df = load_csv(csv_file)
    summary = summarize(df)
    
    assert "mean" in summary.columns
    assert "cnt" in summary.index

def test_filter_rows(tmp_path):
    """Test DataFrame filtering."""
    csv_file = tmp_path / "bikes.csv"
    csv_file.write_text("cnt,temp,season\n985,0.344,1\n801,0.363,1\n1349,0.196,1\n")
    
    df = load_csv(csv_file)
    filtered = filter_rows(df, "cnt >= 900")
    
    assert len(filtered) == 2
    assert filtered["cnt"].min() >= 900

def test_analyze_rentals(tmp_path):
    """Test rental analysis."""
    csv_file = tmp_path / "rental_data.csv"
    csv_file.write_text("cnt,casual,registered\n985,331,654\n801,131,670\n1349,120,1229\n")
    
    df = load_csv(csv_file)
    analysis = analyze_rentals(df)
    
    assert 'total_rentals' in analysis
    assert 'avg_daily_rentals' in analysis
    assert 'casual_vs_registered' in analysis
    assert analysis['total_rentals'] == 3135

def test_analyze_weather_impact(tmp_path):
    """Test weather impact analysis."""
    csv_file = tmp_path / "weather_data.csv"
    csv_file.write_text("cnt,temp,weathersit,hum,windspeed\n985,0.344,2,0.806,0.160\n801,0.363,2,0.696,0.249\n1349,0.196,1,0.437,0.248\n")
    
    df = load_csv(csv_file)
    analysis = analyze_weather_impact(df)
    
    assert 'temperature_correlation' in analysis
    assert 'by_weather_situation' in analysis
    assert isinstance(analysis['temperature_correlation'], float)

def test_analyze_seasonal_patterns(tmp_path):
    """Test seasonal pattern analysis."""
    csv_file = tmp_path / "seasonal_data.csv"
    csv_file.write_text("cnt,season,weekday,workingday,holiday\n985,1,6,0,0\n801,1,0,0,0\n1349,1,1,1,0\n")
    
    df = load_csv(csv_file)
    analysis = analyze_seasonal_patterns(df)
    
    assert 'by_season' in analysis
    assert 'by_weekday' in analysis
    assert 'working_vs_non_working' in analysis