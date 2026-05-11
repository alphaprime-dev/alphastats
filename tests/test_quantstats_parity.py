# mypy: disable-error-code=no-untyped-call
import math
from collections.abc import Callable
from typing import SupportsFloat, cast

import pandas as pd
import polars as pl
import pytest
import quantstats.stats as qs

from alphastats import stats

pytestmark = [
    pytest.mark.filterwarnings("ignore:Mean of empty slice:RuntimeWarning"),
    pytest.mark.filterwarnings("ignore:invalid value encountered in scalar divide:RuntimeWarning"),
]


def returns_series() -> pd.Series:
    return pd.Series(
        [0.01, -0.02, 0.03, -0.01, 0.02, 0.0, 0.015, -0.005, 0.01, -0.015],
        index=pd.date_range("2023-01-02", periods=10, freq="D"),
        name="asset",
    )


def monthly_returns_series() -> pd.Series:
    return pd.Series(
        [0.01, -0.02, 0.03, -0.015, 0.02, -0.01, 0.025, -0.02, 0.015, 0.01],
        index=pd.date_range("2023-01-31", periods=10, freq="ME"),
        name="asset",
    )


def benchmark_series() -> pd.Series:
    return pd.Series(
        [0.005, -0.01, 0.015, -0.005, 0.01, 0.0, 0.007, -0.002, 0.004, -0.006],
        index=pd.date_range("2023-01-02", periods=10, freq="D"),
        name="benchmark",
    )


def to_polars_frame(series: pd.Series, column_name: str = "asset") -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [timestamp.date() for timestamp in series.index],
            column_name: [float(value) for value in series.to_list()],
        }
    )


def to_polars_series(series: pd.Series, column_name: str = "asset") -> pl.Series:
    return pl.Series(column_name, [float(value) for value in series.to_list()])


def assert_close(actual: object, expected: object, rel: float = 1e-9) -> None:
    actual_float = float(cast(SupportsFloat, actual))
    expected_float = float(cast(SupportsFloat, expected))
    if math.isnan(expected_float):
        assert math.isnan(actual_float)
    elif math.isinf(expected_float):
        assert actual_float == expected_float
    else:
        assert actual_float == pytest.approx(expected_float, rel=rel, abs=1e-12)


SERIES_PARITY_CASES: list[
    tuple[str, Callable[[pd.Series], object], Callable[[pl.Series], object]]
] = [
    ("comp", qs.comp, stats.comp),
    ("max_drawdown", qs.max_drawdown, stats.max_drawdown),
    ("sharpe", qs.sharpe, stats.sharpe),
    ("sortino", qs.sortino, stats.sortino),
    ("probabilistic_sharpe_ratio", qs.probabilistic_sharpe_ratio, stats.probabilistic_sharpe_ratio),
    ("psr", qs.probabilistic_sharpe_ratio, stats.psr),
    ("smart_sharpe", qs.smart_sharpe, stats.smart_sharpe),
    ("smart_sortino", qs.smart_sortino, stats.smart_sortino),
    ("adjusted_sortino", qs.adjusted_sortino, stats.adjusted_sortino),
    ("volatility", qs.volatility, stats.volatility),
    ("omega", qs.omega, stats.omega),
    ("skew", qs.skew, stats.skew),
    ("kurtosis", qs.kurtosis, stats.kurtosis),
    ("expected_return", qs.expected_return, stats.expected_return),
    ("geometric_mean", qs.geometric_mean, stats.geometric_mean),
    ("best", qs.best, stats.best),
    ("worst", qs.worst, stats.worst),
    ("avg_return", qs.avg_return, stats.avg_return),
    ("avg_win", qs.avg_win, stats.avg_win),
    ("avg_loss", qs.avg_loss, stats.avg_loss),
    ("win_rate", qs.win_rate, stats.win_rate),
    ("payoff_ratio", qs.payoff_ratio, stats.payoff_ratio),
    ("profit_factor", qs.profit_factor, stats.profit_factor),
    ("cpc_index", qs.cpc_index, stats.cpc_index),
    ("common_sense_ratio", qs.common_sense_ratio, stats.common_sense_ratio),
    ("tail_ratio", qs.tail_ratio, stats.tail_ratio),
    ("outlier_win_ratio", qs.outlier_win_ratio, stats.outlier_win_ratio),
    ("outlier_loss_ratio", qs.outlier_loss_ratio, stats.outlier_loss_ratio),
    ("kelly_criterion", qs.kelly_criterion, stats.kelly_criterion),
    ("risk_of_ruin", qs.risk_of_ruin, stats.risk_of_ruin),
    ("ror", qs.ror, stats.ror),
    ("value_at_risk", qs.value_at_risk, stats.value_at_risk),
    ("var", qs.var, stats.var),
    ("conditional_value_at_risk", qs.conditional_value_at_risk, stats.conditional_value_at_risk),
    ("cvar", qs.cvar, stats.cvar),
    ("expected_shortfall", qs.expected_shortfall, stats.expected_shortfall),
    ("recovery_factor", qs.recovery_factor, stats.recovery_factor),
    ("ulcer_index", qs.ulcer_index, stats.ulcer_index),
    ("serenity_index", qs.serenity_index, stats.serenity_index),
]


@pytest.mark.parametrize(("name", "quantstats_func", "alphastats_func"), SERIES_PARITY_CASES)
def test_series_metrics_match_quantstats(
    name: str,
    quantstats_func: Callable[[pd.Series], object],
    alphastats_func: Callable[[pl.Series], object],
) -> None:
    pandas_returns = returns_series()
    polars_returns = to_polars_series(pandas_returns)

    assert_close(alphastats_func(polars_returns), quantstats_func(pandas_returns))


def test_to_drawdowns_matches_quantstats() -> None:
    pandas_returns = returns_series()
    polars_returns = to_polars_series(pandas_returns)

    actual = stats.to_drawdowns(polars_returns).to_list()
    expected = qs.to_drawdown_series(pandas_returns).to_list()

    assert len(actual) == len(expected)
    for actual_value, expected_value in zip(actual, expected, strict=True):
        assert_close(actual_value, expected_value)


def test_cagr_and_calmar_match_quantstats() -> None:
    pandas_returns = returns_series()
    polars_returns = to_polars_frame(pandas_returns)

    assert_close(stats.cagr(polars_returns)["asset"][0], qs.cagr(pandas_returns))
    assert_close(stats.calmar(polars_returns)["asset"][0], qs.calmar(pandas_returns))


AGGREGATE_PARITY_CASES: list[
    tuple[str, Callable[[pd.Series], object], Callable[[pl.DataFrame], object]]
] = [
    (
        "expected_monthly",
        lambda series: qs.expected_return(series, aggregate="month"),
        stats.expected_monthly,
    ),
    (
        "expected_yearly",
        lambda series: qs.expected_return(series, aggregate="year"),
        stats.expected_yearly,
    ),
    ("best_day", lambda series: qs.best(series, aggregate="day"), stats.best_day),
    ("worst_day", lambda series: qs.worst(series, aggregate="day"), stats.worst_day),
    ("best_month", lambda series: qs.best(series, aggregate="month"), stats.best_month),
    ("worst_month", lambda series: qs.worst(series, aggregate="month"), stats.worst_month),
    ("best_year", lambda series: qs.best(series, aggregate="year"), stats.best_year),
    ("worst_year", lambda series: qs.worst(series, aggregate="year"), stats.worst_year),
    ("avg_up_month", lambda series: qs.avg_win(series, aggregate="month"), stats.avg_up_month),
    ("avg_down_month", lambda series: qs.avg_loss(series, aggregate="month"), stats.avg_down_month),
    ("win_days", lambda series: qs.win_rate(series, aggregate="day"), stats.win_days),
    ("win_month", lambda series: qs.win_rate(series, aggregate="month"), stats.win_month),
    ("win_quarter", lambda series: qs.win_rate(series, aggregate="quarter"), stats.win_quarter),
    ("win_year", lambda series: qs.win_rate(series, aggregate="year"), stats.win_year),
    (
        "gain_to_pain_ratio_1m",
        lambda series: qs.gain_to_pain_ratio(series, resolution="ME"),
        stats.gain_to_pain_ratio_1m,
    ),
]


@pytest.mark.parametrize(("name", "quantstats_func", "alphastats_func"), AGGREGATE_PARITY_CASES)
def test_aggregate_metrics_match_quantstats(
    name: str,
    quantstats_func: Callable[[pd.Series], object],
    alphastats_func: Callable[[pl.DataFrame], object],
) -> None:
    pandas_returns = monthly_returns_series()
    polars_returns = to_polars_frame(pandas_returns)

    alphastats_result = cast(pl.DataFrame, alphastats_func(polars_returns))
    assert_close(alphastats_result["asset"][0], quantstats_func(pandas_returns))


def test_benchmark_metrics_match_quantstats() -> None:
    pandas_returns = returns_series()
    pandas_benchmark = benchmark_series()
    polars_returns = to_polars_frame(pandas_returns)
    polars_benchmark = to_polars_frame(pandas_benchmark, "_benchmark_returns")

    assert_close(
        stats.information_ratio(polars_returns, polars_benchmark)["asset"][0],
        qs.information_ratio(pandas_returns, pandas_benchmark),
    )
    assert_close(
        stats.r_squared(polars_returns, polars_benchmark)["asset"][0],
        qs.r_squared(pandas_returns, pandas_benchmark),
    )
    assert_close(
        stats.r2(polars_returns, polars_benchmark)["asset"][0],
        qs.r2(pandas_returns, pandas_benchmark),
    )
    assert_close(
        stats.treynor_ratio(polars_returns, polars_benchmark)["asset"][0],
        qs.treynor_ratio(pandas_returns, pandas_benchmark),
    )

    quantstats_greeks = qs.greeks(pandas_returns, pandas_benchmark)
    alphastats_greeks = stats.greeks(polars_returns, polars_benchmark)["asset"][0]
    assert_close(alphastats_greeks["alpha"], quantstats_greeks["alpha"])
    assert_close(alphastats_greeks["beta"], quantstats_greeks["beta"])


def test_drawdown_detail_metrics_match_quantstats() -> None:
    pandas_returns = returns_series()
    polars_returns = to_polars_frame(pandas_returns)
    quantstats_details = qs.drawdown_details(qs.to_drawdown_series(pandas_returns))
    longest = cast(pl.DataFrame, stats.longest_drawdown_days(polars_returns))
    avg_days = cast(pl.DataFrame, stats.avg_drawdown_days(polars_returns))
    avg_dd = cast(pl.DataFrame, stats.avg_drawdown(polars_returns))

    assert_close(
        longest["asset"][0],
        quantstats_details["days"].max(),
    )
    assert_close(
        avg_days["asset"][0],
        quantstats_details["days"].mean(),
    )
    assert_close(
        avg_dd["asset"][0],
        quantstats_details["max drawdown"].mean() / 100,
    )


def filtered_comp(series: pd.Series, start: pd.Timestamp | None = None) -> float:
    filtered = series if start is None else series[series.index >= start]
    return float(qs.comp(filtered))


def filtered_cagr(series: pd.Series, years: int | None = None) -> float:
    filtered = (
        series
        if years is None
        else series[series.index >= series.index.max() - pd.DateOffset(years=years)]
    )
    return float(qs.cagr(filtered))


def test_report_window_wrappers_match_equivalent_quantstats_outputs() -> None:
    pandas_returns = monthly_returns_series()
    polars_returns = to_polars_frame(pandas_returns)
    latest = pandas_returns.index.max()

    assert_close(
        stats.mtd(polars_returns)["asset"][0],
        filtered_comp(pandas_returns[pandas_returns.index.month == latest.month]),
    )
    assert_close(
        stats.ytd(polars_returns)["asset"][0],
        filtered_comp(pandas_returns[pandas_returns.index.year == latest.year]),
    )
    assert_close(
        stats.three_month(polars_returns)["asset"][0],
        filtered_comp(pandas_returns, latest - pd.DateOffset(months=3)),
    )
    assert_close(
        stats.six_month(polars_returns)["asset"][0],
        filtered_comp(pandas_returns, latest - pd.DateOffset(months=6)),
    )
    assert_close(
        stats.one_year(polars_returns)["asset"][0],
        filtered_comp(pandas_returns, latest - pd.DateOffset(years=1)),
    )
    assert_close(stats.three_year(polars_returns)["asset"][0], filtered_cagr(pandas_returns, 3))
    assert_close(stats.five_year(polars_returns)["asset"][0], filtered_cagr(pandas_returns, 5))
    assert_close(stats.ten_year(polars_returns)["asset"][0], filtered_cagr(pandas_returns, 10))
    assert_close(stats.all_time(polars_returns)["asset"][0], filtered_cagr(pandas_returns))
