"""
Utility components for epidemic forecasting.

This module contains utility functionality:
- Metrics: Evaluation metrics and KPIs
- Web Sources: Web scraping and data collection
- HTML Generators: Report generation utilities
- Date Utils: Date handling utilities
"""

from .metrics import (
    smape, crps_gaussian, interval_coverage, mae, 
    peak_metrics, kpi_exceed_probs
)
from .web_sources import (
    bing_news_search, serpapi_news_search, serpapi_google_news_search,
    serpapi_google_web_search, fetch_official_stats_signals, fetch_web_signals
)
from .html_generators import (
    build_forecast_html, build_holdout_html
)
from .date_utils import iso, dt, weekly_dates_between

__all__ = [
    # Metrics
    "smape", "crps_gaussian", "interval_coverage", "mae",
    "peak_metrics", "kpi_exceed_probs",
    
    # Web Sources
    "bing_news_search", "serpapi_news_search", "serpapi_google_news_search",
    "serpapi_google_web_search", "fetch_official_stats_signals", "fetch_web_signals",
    
    # HTML Generators
    "build_forecast_html", "build_holdout_html",
    
    # Date Utils
    "iso", "dt", "weekly_dates_between",
]
