"""Screener view: run the screen over the S&P 500 + watchlist and rank results."""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import streamlit as st

from src.config import Settings
from src.data.cache import SQLiteCache
from src.data.universe import UniverseResult, load_sp500_universe
from src.data.yahoo_client import YahooFinanceClient
from src.screener.engine import FilterConfig, ScreenerEngine
from src.ui.files import watchlist_tickers
from src.ui.formatting import apply_formatters, dollars, integer, money, percent, score


def _sector_summary(results: pd.DataFrame) -> pd.DataFrame:
    if results is None or results.empty or 'Sector' not in results.columns:
        return pd.DataFrame(
            columns=['Sector', 'Candidates', 'Avg Confidence', 'Avg Rank Score', 'Avg Sector Trend']
        )
    grouped = (
        results.groupby('Sector', dropna=False)
        .agg(
            Candidates=('Ticker', 'count'),
            Avg_Confidence=('Confidence', 'mean'),
            Avg_Rank=('Rank Score', 'mean'),
            Avg_Sector_Trend=('Sector Trend', 'mean'),
        )
        .reset_index()
    )
    grouped = grouped.sort_values(['Avg_Rank', 'Avg_Confidence'], ascending=False)
    grouped.columns = [
        'Sector',
        'Candidates',
        'Avg Confidence',
        'Avg Rank Score',
        'Avg Sector Trend',
    ]
    return grouped


def _screen_quality_summary(results: pd.DataFrame) -> pd.DataFrame:
    """Summarize broad tape health, sector leadership, and setup quality."""
    if results is None or results.empty:
        return pd.DataFrame(columns=['Metric', 'Value', 'Detail'])

    avg_confidence = float(results['Confidence'].mean()) if 'Confidence' in results.columns else 0.0
    avg_rank = float(results['Rank Score'].mean()) if 'Rank Score' in results.columns else 0.0
    avg_sector_trend = (
        float(results['Sector Trend'].mean()) if 'Sector Trend' in results.columns else 0.0
    )
    avg_rr = float(results['R/R'].mean()) if 'R/R' in results.columns else 0.0

    breadth = (
        'Strong' if avg_sector_trend > 0.02 else 'Moderate' if avg_sector_trend >= -0.01 else 'Weak'
    )
    sector_lead = 'Strong' if avg_rank >= 75 else 'Moderate' if avg_rank >= 60 else 'Weak'
    quality = (
        'Strong'
        if avg_confidence >= 75 and avg_rr >= 2.0
        else 'Moderate'
        if avg_confidence >= 60
        else 'Weak'
    )

    return pd.DataFrame(
        [
            {
                'Metric': 'Market Breadth',
                'Value': breadth,
                'Detail': f'Avg sector trend: {avg_sector_trend:.2%}',
            },
            {
                'Metric': 'Sector Leadership',
                'Value': sector_lead,
                'Detail': f'Avg rank score: {avg_rank:.1f}',
            },
            {
                'Metric': 'Setup Quality',
                'Value': quality,
                'Detail': f'Avg confidence: {avg_confidence:.1f} | avg R/R: {avg_rr:.2f}',
            },
        ]
    )


def _filter_results_by_sector(
    results: pd.DataFrame, sectors: tuple[str, ...] | list[str] | None
) -> pd.DataFrame:
    if results is None or results.empty or 'Sector' not in results.columns:
        return results.copy() if isinstance(results, pd.DataFrame) else pd.DataFrame()
    selected = tuple(str(item).strip() for item in (sectors or ()) if str(item).strip())
    if not selected:
        return results.copy()
    return results[results['Sector'].astype(str).isin(selected)].copy()


# Columns shown in the screener table, in display order.
_DISPLAY_COLUMNS = (
    'Ticker',
    'Company Name',
    'Setup',
    'Confidence',
    'Rank Score',
    'Entry',
    'Stop',
    'Target',
    'R/R',
    'Reward %',
    'Risk %',
    'Trend Score',
    'Rel Volume',
    'Beta',
    'Return 3M',
    'ATR %',
    'Dist 200D %',
    'Dollar ADV',
    'Div Yield',
    'Sector',
    'Sector Trend',
    'Market Cap',
    'Market Context',
)

_FORMATTERS = {
    'Confidence': integer,
    'Rank Score': score,
    'R/R': score,
    'Entry': money,
    'Stop': money,
    'Target': money,
    'Reward %': percent,
    'Risk %': percent,
    'Trend Score': score,
    'Rel Volume': score,
    'Beta': score,
    'Return 3M': percent,
    'ATR %': percent,
    'Dist 200D %': percent,
    'Div Yield': percent,
    'Dollar ADV': dollars,
    'Sector Trend': percent,
    'Market Cap': money,
}


def render_screener(
    cache: SQLiteCache,
    client: YahooFinanceClient,
    settings: Settings,
    engine: ScreenerEngine,
    config: FilterConfig,
) -> None:
    st.subheader('Screener')
    st.caption(
        'Ranked high-conviction technical setups from the S&P 500 plus your '
        'watchlist. Entry / Stop / Target are structural, data-derived levels. '
        'Gates are intentionally tight, so a short list is expected.'
    )
    if st.button('Run screen', type='primary'):
        _run_screen(cache, client, engine, config)

    results = st.session_state.get('screen_results')
    if results is None:
        st.info('Click "Run screen" to scan the S&P 500 + watchlist for fresh setups.')
        return

    st.caption(f'Last run: {st.session_state.get("screen_at", "-")}')
    if results.empty:
        st.success('No candidates cleared the gates \u2014 nothing actionable right now.')
        return

    quality = _screen_quality_summary(results)
    if not quality.empty:
        st.subheader('Screen quality summary')
        metric_cols = st.columns(len(quality))
        for idx, row in enumerate(quality.itertuples(index=False)):
            with metric_cols[idx]:
                st.metric(row.Metric, row.Value, row.Detail)

    summary = st.columns(3)
    with summary[0]:
        st.metric('Candidates', len(results))
    with summary[1]:
        best_conf = float(results['Confidence'].max()) if 'Confidence' in results.columns else 0.0
        st.metric('Best Confidence', f'{best_conf:.0f}')
    with summary[2]:
        market_context = (
            results['Market Context'].mode().iloc[0]
            if 'Market Context' in results.columns
            else 'Neutral'
        )
        st.metric('Tape', str(market_context))

    sector_options = []
    if 'Sector' in results.columns:
        sector_options = sorted({str(value) for value in results['Sector'].dropna().unique()})

    filtered_results = results
    if sector_options:
        selected_sectors = st.multiselect(
            'Filter by sector',
            options=sector_options,
            default=sector_options,
            key='sector_filter',
            help='Focus the leaderboard and table on the sectors currently showing the strongest leadership.',
        )
        filtered_results = _filter_results_by_sector(results, selected_sectors)

    summary = _sector_summary(filtered_results)
    if not summary.empty:
        st.subheader('Sector leadership')
        st.dataframe(
            apply_formatters(
                summary.head(5),
                {'Avg Confidence': score, 'Avg Rank Score': score, 'Avg Sector Trend': percent},
            ),
            hide_index=True,
            width='stretch',
        )

    display = filtered_results.reindex(
        columns=[c for c in _DISPLAY_COLUMNS if c in filtered_results.columns]
    )
    st.dataframe(apply_formatters(display, _FORMATTERS), width='stretch', hide_index=True)
    st.caption(
        f'{len(filtered_results)} candidate(s) shown after sector filtering ({len(results)} total).'
    )


def _run_screen(
    cache: SQLiteCache,
    client: YahooFinanceClient,
    engine: ScreenerEngine,
    config: FilterConfig,
) -> None:
    universe = load_sp500_universe(cache)
    tickers = list(dict.fromkeys([*universe.tickers, *watchlist_tickers()]))
    full = UniverseResult(tickers=tickers, companies=dict(universe.companies))
    with st.spinner(f'Screening {len(tickers)} names\u2026'):
        results = engine.screen(full, config=config)
    if not results.empty:
        results = results.sort_values('Rank Score', ascending=False).reset_index(drop=True)
    st.session_state['screen_results'] = results
    st.session_state['screen_at'] = datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')
