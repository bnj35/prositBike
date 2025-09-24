"""
Usage (inside Docker container):
    python -m bike.graphs /data/day.csv --hist --reg --x temp --y cnt
"""
from __future__ import annotations

import argparse
from pathlib import Path
import os
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def ensure_plots_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def load_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")
    df = pd.read_csv(p)
    if 'dteday' in df.columns:
        df['dteday'] = pd.to_datetime(df['dteday'])
    return df


def plot_histograms(df: pd.DataFrame, columns: Optional[list[str]] = None, out_dir: Path = Path('plots')) -> list[Path]:
    """Plot histograms for the specified columns (or sensible defaults).

    Returns list of generated file paths.
    """
    ensure_plots_dir(out_dir)
    files = []

    if columns is None:
        # sensible defaults
        columns = [c for c in ['cnt', 'temp', 'hum', 'windspeed'] if c in df.columns]

    for col in columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        fname = out_dir / f'hist_{col}.png'
        plt.tight_layout()
        plt.savefig(fname)
        plt.close()
        files.append(fname)

    return files


def plot_regression(df: pd.DataFrame, x: str, y: str, out_dir: Path = Path('plots')) -> Optional[Path]:
    """Create a regression scatter plot with linear fit between x and y.

    Returns the generated file path or None if columns missing.
    """
    if x not in df.columns or y not in df.columns:
        return None

    ensure_plots_dir(out_dir)
    plt.figure(figsize=(8, 6))
    sns.regplot(x=x, y=y, data=df, scatter_kws={'s': 20, 'alpha': 0.6}, line_kws={'color': 'red'})
    plt.title(f'Regression: {y} ~ {x}')
    plt.xlabel(x)
    plt.ylabel(y)
    fname = out_dir / f'reg_{y}_vs_{x}.png'
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    return fname


def plot_time_series(df: pd.DataFrame, out_dir: Path = Path('plots')) -> Optional[Path]:
    """Plot time series of total rentals over time (daily)."""
    if 'dteday' not in df.columns or 'cnt' not in df.columns:
        return None

    ensure_plots_dir(out_dir)
    plt.figure(figsize=(12, 5))
    sns.lineplot(x='dteday', y='cnt', data=df)
    plt.title('Daily Rentals Over Time')
    plt.xlabel('Date')
    plt.ylabel('Rentals (cnt)')
    fname = out_dir / 'ts_daily_rentals.png'
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    return fname


def plot_monthly_average(df: pd.DataFrame, out_dir: Path = Path('plots')) -> Optional[Path]:
    """Plot average rentals per month (bar chart)."""
    if 'mnth' not in df.columns or 'cnt' not in df.columns:
        return None

    ensure_plots_dir(out_dir)
    monthly = df.groupby('mnth')['cnt'].mean().reindex(range(1, 13))
    plt.figure(figsize=(10, 5))
    sns.barplot(x=monthly.index - 1, y=monthly.values, palette='viridis')
    plt.title('Average Daily Rentals by Month')
    plt.xlabel('Month')
    plt.ylabel('Average Rentals')
    plt.xticks(range(0, 12), ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    fname = out_dir / 'avg_rentals_by_month.png'
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    return fname


def plot_weekday_distribution(df: pd.DataFrame, out_dir: Path = Path('plots')) -> Optional[Path]:
    """Plot average rentals by weekday."""
    if 'weekday' not in df.columns or 'cnt' not in df.columns:
        return None

    ensure_plots_dir(out_dir)
    weekday = df.groupby('weekday')['cnt'].mean().reindex(range(0,7))
    plt.figure(figsize=(8, 4))
    sns.barplot(x=weekday.index, y=weekday.values, palette='magma')
    plt.title('Average Rentals by Weekday')
    plt.xlabel('Weekday (0=Sunday)')
    plt.ylabel('Average Rentals')
    plt.xticks(range(0,7), ['Sun','Mon','Tue','Wed','Thu','Fri','Sat'])
    fname = out_dir / 'avg_rentals_by_weekday.png'
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    return fname


def plot_holiday_comparison(df: pd.DataFrame, out_dir: Path = Path('plots')) -> Optional[Path]:
    """Compare rental distributions on holidays vs regular days."""
    if 'holiday' not in df.columns or 'cnt' not in df.columns:
        return None

    ensure_plots_dir(out_dir)
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='holiday', y='cnt', data=df)
    plt.title('Rentals: Holidays vs Regular Days')
    plt.xlabel('Holiday (0=No, 1=Yes)')
    plt.ylabel('Rentals')
    fname = out_dir / 'box_holiday_vs_regular.png'
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    return fname


def plot_month_boxplots(df: pd.DataFrame, out_dir: Path = Path('plots')) -> Optional[Path]:
    """Boxplots of daily rentals for each month to show variability and outliers."""
    if 'mnth' not in df.columns or 'cnt' not in df.columns:
        return None

    ensure_plots_dir(out_dir)
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='mnth', y='cnt', data=df, palette='coolwarm')
    plt.title('Monthly Distribution of Daily Rentals')
    plt.xlabel('Month')
    plt.ylabel('Daily Rentals')
    plt.xticks(range(0,12), ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'])
    fname = out_dir / 'box_monthly_rentals.png'
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    return fname


def plot_month_weekday_heatmap(df: pd.DataFrame, out_dir: Path = Path('plots')) -> Optional[Path]:
    """Heatmap of average rentals for each month-weekday combination."""
    if 'mnth' not in df.columns or 'weekday' not in df.columns or 'cnt' not in df.columns:
        return None

    ensure_plots_dir(out_dir)
    pivot = df.pivot_table(index='mnth', columns='weekday', values='cnt', aggfunc='mean').reindex(index=range(1,13), columns=range(0,7))
    plt.figure(figsize=(10, 8))
    sns.heatmap(pivot, annot=True, fmt='.0f', cmap='YlGnBu')
    plt.title('Average Rentals: Month vs Weekday')
    plt.xlabel('Weekday (0=Mon)')
    plt.ylabel('Month')
    fname = out_dir / 'heatmap_month_weekday.png'
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()
    return fname


def main():
    parser = argparse.ArgumentParser(description='Generate plots from bike sharing CSV data')
    parser.add_argument('csv', help='Path to CSV file')
    parser.add_argument('--hist', action='store_true', help='Generate histograms')
    parser.add_argument('--hist-cols', nargs='+', help='Columns for histograms')
    parser.add_argument('--reg', action='store_true', help='Generate regression plot')
    parser.add_argument('--x', help='X column for regression (default: temp)')
    parser.add_argument('--y', help='Y column for regression (default: cnt)')
    parser.add_argument('--out', default='plots', help='Output directory for plots')
    parser.add_argument('--presentation', action='store_true', help='Generate presentation-ready plots (time series, monthly, weekday, holiday, boxplots, heatmap)')
    args = parser.parse_args()

    out_dir = Path(args.out)

    try:
        df = load_csv(args.csv)
    except Exception as e:
        print(f"Failed to load CSV: {e}")
        return 1

    generated = []
    if args.hist:
        cols = args.hist_cols if args.hist_cols else None
        files = plot_histograms(df, columns=cols, out_dir=out_dir)
        print(f"Generated histograms: {[str(p) for p in files]}")
        generated.extend(files)

    if args.reg:
        x = args.x if args.x else 'temp'
        y = args.y if args.y else 'cnt'
        file = plot_regression(df, x, y, out_dir=out_dir)
        if file:
            print(f"Generated regression plot: {file}")
            generated.append(file)
        else:
            print(f"Could not generate regression: missing column {x} or {y}")

    if args.presentation:
        print('Generating presentation plots...')
        ts = plot_time_series(df, out_dir=out_dir)
        ma = plot_monthly_average(df, out_dir=out_dir)
        wd = plot_weekday_distribution(df, out_dir=out_dir)
        hol = plot_holiday_comparison(df, out_dir=out_dir)
        box = plot_month_boxplots(df, out_dir=out_dir)
        heat = plot_month_weekday_heatmap(df, out_dir=out_dir)

        created = {k: str(v) for k, v in [('time_series', ts), ('monthly_avg', ma), ('weekday', wd), ('holiday_box', hol), ('monthly_box', box), ('heatmap', heat)] if v}
        print('Created presentation plots:')
        for k, v in created.items():
            print(f"  - {k}: {v}")
        generated.extend(created.values())

    if not any([args.hist, args.reg]):
        print('No plots requested. Use --hist and/or --reg')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
