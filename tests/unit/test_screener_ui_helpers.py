import pandas as pd

from src.ui.screener import (
    _filter_results_by_sector,
    _screen_quality_summary,
    _sector_summary,
)


def test_sector_summary_ranks_leaders_by_quality() -> None:
    results = pd.DataFrame(
        {
            'Ticker': ['A', 'B', 'C'],
            'Sector': ['Tech', 'Tech', 'Energy'],
            'Confidence': [80, 60, 40],
            'Rank Score': [90, 70, 50],
            'Sector Trend': [0.08, 0.03, -0.02],
        }
    )

    summary = _sector_summary(results)

    assert list(summary['Sector']) == ['Tech', 'Energy']
    assert summary.iloc[0]['Avg Confidence'] == 70.0
    assert summary.iloc[0]['Avg Rank Score'] == 80.0


def test_filter_results_by_sector_keeps_only_selected_groups() -> None:
    results = pd.DataFrame(
        {
            'Ticker': ['A', 'B', 'C'],
            'Sector': ['Tech', 'Health Care', 'Energy'],
            'Confidence': [70, 65, 55],
        }
    )

    filtered = _filter_results_by_sector(results, ('Tech',))

    assert list(filtered['Ticker']) == ['A']
    assert list(filtered['Sector']) == ['Tech']


def test_screen_quality_summary_reports_market_health() -> None:
    results = pd.DataFrame(
        {
            'Ticker': ['A', 'B', 'C'],
            'Sector': ['Tech', 'Tech', 'Energy'],
            'Confidence': [82, 76, 58],
            'Rank Score': [85, 72, 50],
            'Sector Trend': [0.12, 0.09, -0.03],
            'R/R': [2.5, 2.2, 1.2],
        }
    )

    summary = _screen_quality_summary(results)

    assert set(summary['Metric']) >= {'Market Breadth', 'Sector Leadership', 'Setup Quality'}
    assert summary[summary['Metric'] == 'Setup Quality']['Value'].iloc[0] in {'Strong', 'Moderate'}
