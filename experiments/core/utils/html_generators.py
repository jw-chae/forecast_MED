"""
Unified HTML generation utilities for epidemic forecasting reports.
"""

import json
from typing import Dict, List, Any, Optional


def build_forecast_html(dates: List[str], values: List[float], 
                       future_dates: List[str], forecast: dict, disease: str) -> str:
    """Generate HTML for basic forecast visualization."""
    plotly_cdn = "https://cdn.plot.ly/plotly-2.27.1.min.js"
    
    # Prepare data for JavaScript
    data_dict = {
        'dates': dates, 
        'values': values, 
        'future_dates': future_dates, 
        'forecast': forecast
    }
    data_json = json.dumps(data_dict)
    
    html = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="utf-8"/>
    <title>Forecast Report - {disease}</title>
    <script src="{plotly_cdn}"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; }}
        #chart {{ width: 100%; height: 600px; }}
    </style>
</head>
<body>
    <h1>전염병 예측 리포트 - {disease}</h1>
    <div id="chart"></div>
    <script>
        const data = {data_json};
        const traces = [
            {{x: data.dates, y: data.values, name: '실측값', mode: 'lines+markers', line: {{color: '#1f77b4'}}}},
            {{x: data.future_dates, y: [data.forecast.mean] * data.future_dates.length, name: '예측값', mode: 'lines', line: {{color: '#ff7f0e', dash: 'dash'}}}}
        ];
        Plotly.newPlot('chart', traces, {{title: '{disease} 전염병 예측', xaxis: {{title: '날짜'}}, yaxis: {{title: '발병자 수'}}}});
    </script>
</body>
</html>
"""
    return html


def build_holdout_html(dates_hist: List[str], values_hist: List[float], 
                      dates_target: List[str], values_target: List[float],
                      future_dates: List[str], forecast: dict, disease: str,
                      train_until: str, end_date: str) -> str:
    """Generate HTML for holdout validation results."""
    plotly_cdn = "https://cdn.plot.ly/plotly-2.27.1.min.js"
    
    html = f"""
<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="utf-8"/>
    <title>Holdout Validation - {disease}</title>
    <script src="{plotly_cdn}"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 20px; }}
        #chart {{ width: 100%; height: 600px; }}
    </style>
</head>
<body>
    <h1>홀드아웃 검증 결과 - {disease}</h1>
    <div id="chart"></div>
    <script>
        const traces = [
            {{x: {json.dumps(dates_hist)}, y: {json.dumps(values_hist)}, name: '학습 데이터', mode: 'lines', line: {{color: '#999'}}}},
            {{x: {json.dumps(dates_target)}, y: {json.dumps(values_target)}, name: '실측값', mode: 'lines+markers', line: {{color: '#1f77b4'}}}},
            {{x: {json.dumps(future_dates)}, y: [{forecast.get('mean', 0)}] * {len(future_dates)}, name: '예측값', mode: 'lines', line: {{color: '#ff7f0e', dash: 'dash'}}}}
        ];
        Plotly.newPlot('chart', traces, {{title: '{disease} 홀드아웃 검증', xaxis: {{title: '날짜'}}, yaxis: {{title: '발병자 수'}}}});
    </script>
</body>
</html>
"""
    return html

