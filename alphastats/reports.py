from __future__ import annotations

import math
from collections.abc import Callable
from datetime import date, datetime
from typing import Any, cast

import polars as pl
import polars.selectors as cs

from alphastats import stats
from alphastats._utils import (
    RETURNS_COLUMNS_SELECTOR,
    get_temporal_column,
)

ReturnFrame = pl.DataFrame | pl.LazyFrame
Returns = pl.Series | ReturnFrame
MetricFunc = Callable[[pl.Series], Any]


def metrics(
    returns: Returns,
    benchmark: Returns | None = None,
    rf: float = 0.0,
    display: bool = True,
    mode: str = "basic",
    sep: bool = False,
    compounded: bool = True,
    periods_per_year: int = 252,
    prepare_returns: bool = True,
    match_dates: bool = True,
    **kwargs: Any,
) -> pl.DataFrame | None:
    """Calculate a QuantStats-style performance metrics table."""
    del prepare_returns

    report = _prepare_report_frame(
        returns,
        benchmark,
        match_dates=match_dates,
        strategy_title=kwargs.get("strategy_title", "Strategy"),
        benchmark_title=kwargs.get("benchmark_title", "Benchmark"),
    )
    pct = 100 if display or "internal" in kwargs or kwargs.get("as_pct", False) else 1
    full = mode.lower() == "full"

    rows: list[tuple[str, dict[str, Any]]] = []

    def add(label: str, values: dict[str, Any]) -> None:
        rows.append((label, values))

    def add_sep() -> None:
        if sep:
            rows.append(("", dict.fromkeys(report.output_names, "")))

    add(
        "Start Period",
        {name: _format_date(report.start_dates.get(name)) for name in report.output_names},
    )
    add(
        "End Period",
        {name: _format_date(report.end_dates.get(name)) for name in report.output_names},
    )
    add("Risk-Free Rate", dict.fromkeys(report.output_names, rf * pct))
    add("Time in Market", _series_metric(report, stats.exposure, multiplier=pct))
    add_sep()

    if compounded:
        add("Cumulative Return", _series_metric(report, stats.comp, multiplier=pct))
    else:
        add("Total Return", {name: _sum(series) * pct for name, series in report.series.items()})
    add(
        "CAGR﹪",
        _series_cagr(
            report, rf=rf, compounded=compounded, periods=periods_per_year, multiplier=pct
        ),
    )
    add_sep()

    add(
        "Sharpe",
        _series_metric(
            report, lambda series: stats.sharpe(series, rf=rf, periods=periods_per_year)
        ),
    )
    add(
        "Prob. Sharpe Ratio",
        _series_metric(
            report,
            lambda series: stats.probabilistic_sharpe_ratio(
                series, rf=rf, periods=periods_per_year, annualize=False
            ),
            multiplier=pct,
        ),
    )
    if full:
        add(
            "Smart Sharpe",
            _series_metric(
                report, lambda series: stats.smart_sharpe(series, rf=rf, periods=periods_per_year)
            ),
        )
    add(
        "Sortino",
        _series_metric(
            report, lambda series: stats.sortino(series, rf=rf, periods=periods_per_year)
        ),
    )
    if full:
        add(
            "Smart Sortino",
            _series_metric(
                report, lambda series: stats.smart_sortino(series, rf=rf, periods=periods_per_year)
            ),
        )
    add(
        "Sortino/√2",
        _series_metric(
            report, lambda series: stats.adjusted_sortino(series, rf=rf, periods=periods_per_year)
        ),
    )
    if full:
        add(
            "Smart Sortino/√2",
            _series_metric(
                report,
                lambda series: stats.smart_adjusted_sortino(
                    series, rf=rf, periods=periods_per_year
                ),
            ),
        )
    add("Omega", _series_metric(report, stats.omega))
    add_sep()

    add("Max Drawdown", _series_metric(report, stats.max_drawdown, multiplier=pct))
    add("Longest DD Days", _series_metric(report, stats.longest_drawdown_days))

    if full:
        add_sep()
        add(
            "Volatility (ann.)",
            _series_metric(
                report,
                lambda series: stats.volatility(series, periods=periods_per_year),
                multiplier=pct,
            ),
        )
        if benchmark is not None:
            add(
                "R^2",
                _benchmark_metric(report, lambda series, bench: stats.r_squared(series, bench)),
            )
            add(
                "Information Ratio",
                _benchmark_metric(
                    report, lambda series, bench: stats.information_ratio(series, bench)
                ),
            )
        add(
            "Calmar",
            _series_cagr(
                report,
                rf=0.0,
                compounded=compounded,
                periods=periods_per_year,
                multiplier=1,
                divide_by_mdd=True,
            ),
        )
        add("Skew", _series_metric(report, stats.skew))
        add("Kurtosis", _series_metric(report, stats.kurtosis))
        add("Expected Daily", _series_metric(report, stats.expected_daily, multiplier=pct))
        add("Expected Monthly", _monthly_metric(report, stats.expected_monthly, multiplier=pct))
        add("Expected Yearly", _monthly_metric(report, stats.expected_yearly, multiplier=pct))
        add("Kelly Criterion", _series_metric(report, stats.kelly_criterion, multiplier=pct))
        add("Risk of Ruin", _series_metric(report, stats.risk_of_ruin))
        add("Daily Value-at-Risk", _series_metric(report, stats.value_at_risk, multiplier=pct))
        add(
            "Expected Shortfall (cVaR)",
            _series_metric(report, stats.conditional_value_at_risk, multiplier=pct),
        )
        add("Max Consecutive Wins", _series_metric(report, stats.consecutive_wins))
        add("Max Consecutive Losses", _series_metric(report, stats.consecutive_losses))

    add_sep()
    add(
        "Gain/Pain Ratio",
        _series_metric(report, lambda series: stats.gain_to_pain_ratio(series, rf=rf)),
    )
    add(
        "Gain/Pain (1M)",
        _monthly_metric(report, lambda frame: stats.gain_to_pain_ratio_1m(frame, rf=rf)),
    )
    add_sep()
    add("Payoff Ratio", _series_metric(report, stats.payoff_ratio))
    add("Profit Factor", _series_metric(report, stats.profit_factor))
    add("Common Sense Ratio", _series_metric(report, stats.common_sense_ratio))
    add("CPC Index", _series_metric(report, stats.cpc_index))
    add("Tail Ratio", _series_metric(report, stats.tail_ratio))
    add("Outlier Win Ratio", _series_metric(report, stats.outlier_win_ratio))
    add("Outlier Loss Ratio", _series_metric(report, stats.outlier_loss_ratio))
    add_sep()

    add("MTD", _window_metric(report, stats.mtd, multiplier=pct))
    add("3M", _window_metric(report, stats.three_month, multiplier=pct))
    add("6M", _window_metric(report, stats.six_month, multiplier=pct))
    add("YTD", _window_metric(report, stats.ytd, multiplier=pct))
    add("1Y", _window_metric(report, stats.one_year, multiplier=pct))
    add(
        "3Y (ann.)",
        _window_metric(
            report, lambda frame: stats.three_year(frame, periods=periods_per_year), multiplier=pct
        ),
    )
    add(
        "5Y (ann.)",
        _window_metric(
            report, lambda frame: stats.five_year(frame, periods=periods_per_year), multiplier=pct
        ),
    )
    add(
        "10Y (ann.)",
        _window_metric(
            report, lambda frame: stats.ten_year(frame, periods=periods_per_year), multiplier=pct
        ),
    )
    add(
        "All-time (ann.)",
        _series_cagr(
            report, rf=0.0, compounded=compounded, periods=periods_per_year, multiplier=pct
        ),
    )

    if full:
        add_sep()
        add("Best Day", _series_metric(report, stats.best_day, multiplier=pct))
        add("Worst Day", _series_metric(report, stats.worst_day, multiplier=pct))
        add("Best Month", _monthly_metric(report, stats.best_month, multiplier=pct))
        add("Worst Month", _monthly_metric(report, stats.worst_month, multiplier=pct))
        add("Best Year", _monthly_metric(report, stats.best_year, multiplier=pct))
        add("Worst Year", _monthly_metric(report, stats.worst_year, multiplier=pct))

    add_sep()
    add("Avg. Drawdown", _series_metric(report, stats.avg_drawdown, multiplier=pct))
    add("Avg. Drawdown Days", _series_metric(report, stats.avg_drawdown_days))
    add("Recovery Factor", _series_metric(report, stats.recovery_factor))
    add("Ulcer Index", _series_metric(report, stats.ulcer_index))
    add(
        "Serenity Index", _series_metric(report, lambda series: stats.serenity_index(series, rf=rf))
    )

    if full:
        add_sep()
        add("Avg. Up Month", _monthly_metric(report, stats.avg_up_month, multiplier=pct))
        add("Avg. Down Month", _monthly_metric(report, stats.avg_down_month, multiplier=pct))
        add("Win Days", _series_metric(report, stats.win_days, multiplier=pct))
        add("Win Month", _monthly_metric(report, stats.win_month, multiplier=pct))
        add("Win Quarter", _monthly_metric(report, stats.win_quarter, multiplier=pct))
        add("Win Year", _monthly_metric(report, stats.win_year, multiplier=pct))
        if benchmark is not None:
            add_sep()
            add(
                "Beta",
                _benchmark_metric(report, lambda series, bench: _greek(series, bench, "beta")),
            )
            add(
                "Alpha",
                _benchmark_metric(report, lambda series, bench: _greek(series, bench, "alpha")),
            )
            add(
                "Correlation",
                _benchmark_metric(
                    report, lambda series, bench: stats.correlation(series, bench), multiplier=pct
                ),
            )
            add(
                "Treynor Ratio",
                _benchmark_metric(
                    report,
                    lambda series, bench: stats.treynor_ratio(series, bench, rf=rf),
                    multiplier=pct,
                ),
            )

    result = _build_output(rows, report.output_names)
    if display:
        print(result)
        return None
    return result


class _ReportFrame:
    def __init__(
        self,
        frame: pl.DataFrame,
        series: dict[str, pl.Series],
        output_names: list[str],
        benchmark_name: str | None,
        dates: list[Any] | None,
    ) -> None:
        self.frame = frame
        self.series = series
        self.output_names = output_names
        self.benchmark_name = benchmark_name
        self.dates = dates
        self.start_dates = _period_dates(frame, output_names, dates, first=True)
        self.end_dates = _period_dates(frame, output_names, dates, first=False)


def _prepare_report_frame(
    returns: Returns,
    benchmark: Returns | None,
    *,
    match_dates: bool,
    strategy_title: str | list[str],
    benchmark_title: str,
) -> _ReportFrame:
    frame = _to_frame(returns)
    temporal_name = _temporal_name(frame)
    return_names = list(cs.expand_selector(frame.lazy(), RETURNS_COLUMNS_SELECTOR))
    if not return_names:
        msg = "`returns` must contain at least one numeric return column"
        raise ValueError(msg)

    if len(return_names) == 1 and isinstance(strategy_title, str):
        output_names = [strategy_title]
    elif isinstance(strategy_title, list):
        output_names = strategy_title
    else:
        output_names = return_names

    if len(output_names) != len(return_names):
        msg = "`strategy_title` must match the number of strategy return columns"
        raise ValueError(msg)

    selected = [
        pl.col(name).alias(output) for name, output in zip(return_names, output_names, strict=True)
    ]
    if temporal_name:
        selected.insert(0, pl.col(temporal_name))
    frame = frame.select(selected)

    benchmark_name = None
    if benchmark is not None:
        benchmark_frame = _to_frame(benchmark)
        benchmark_temporal = _temporal_name(benchmark_frame)
        benchmark_cols = list(cs.expand_selector(benchmark_frame.lazy(), RETURNS_COLUMNS_SELECTOR))
        if len(benchmark_cols) != 1:
            msg = "`benchmark` must contain exactly one numeric return column"
            raise ValueError(msg)
        benchmark_name = benchmark_title
        benchmark_frame = benchmark_frame.select(
            *([pl.col(benchmark_temporal)] if benchmark_temporal else []),
            pl.col(benchmark_cols[0]).alias(benchmark_name),
        )
        if temporal_name and benchmark_temporal:
            if match_dates:
                frame = frame.join(
                    benchmark_frame,
                    left_on=temporal_name,
                    right_on=benchmark_temporal,
                    how="inner",
                )
            else:
                frame = frame.join(
                    benchmark_frame,
                    left_on=temporal_name,
                    right_on=benchmark_temporal,
                    how="left",
                )
        else:
            frame = pl.concat([frame, benchmark_frame.select(benchmark_name)], how="horizontal")
        output_names = [benchmark_name, *output_names]

    frame = frame.fill_nan(None)
    dates = frame[temporal_name].to_list() if temporal_name else None
    series = {name: frame[name].fill_null(0) for name in output_names}
    return _ReportFrame(frame, series, output_names, benchmark_name, dates)


def _to_frame(returns: Returns) -> pl.DataFrame:
    if isinstance(returns, pl.Series):
        return pl.DataFrame(returns)
    if isinstance(returns, pl.LazyFrame):
        return returns.collect()
    return returns


def _temporal_name(frame: pl.DataFrame) -> str | None:
    temporal = get_temporal_column(frame.lazy())
    if temporal is None:
        return None
    return temporal.meta.output_name()


def _series_metric(
    report: _ReportFrame, func: MetricFunc, multiplier: float = 1.0
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name, series in report.series.items():
        values[name] = _number(func(series), multiplier=multiplier)
    return values


def _benchmark_metric(
    report: _ReportFrame,
    func: Callable[[pl.Series, pl.Series], Any],
    multiplier: float = 1.0,
) -> dict[str, Any]:
    values = dict.fromkeys(report.output_names, "-")
    if report.benchmark_name is None:
        return values
    benchmark = report.series[report.benchmark_name]
    for name, series in report.series.items():
        if name == report.benchmark_name:
            continue
        values[name] = _number(func(series, benchmark), multiplier=multiplier)
    return values


def _monthly_metric(
    report: _ReportFrame,
    func: Callable[[ReturnFrame], Any],
    multiplier: float = 1.0,
) -> dict[str, Any]:
    if report.dates is None:
        return dict.fromkeys(report.output_names, "-")
    values: dict[str, Any] = {}
    for name in report.output_names:
        frame = pl.DataFrame({"date": report.dates, name: report.series[name]})
        values[name] = _frame_value(func(frame), name, multiplier=multiplier)
    return values


def _window_metric(
    report: _ReportFrame,
    func: Callable[[ReturnFrame], Any],
    multiplier: float = 1.0,
) -> dict[str, Any]:
    if report.dates is None:
        return dict.fromkeys(report.output_names, "-")
    values: dict[str, Any] = {}
    for name in report.output_names:
        frame = pl.DataFrame({"date": report.dates, name: report.series[name]})
        values[name] = _frame_value(func(frame), name, multiplier=multiplier)
    return values


def _series_cagr(
    report: _ReportFrame,
    *,
    rf: float,
    compounded: bool,
    periods: int,
    multiplier: float,
    divide_by_mdd: bool = False,
) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name, series in report.series.items():
        cagr_value = _cagr(series, rf=rf, compounded=compounded, periods=periods)
        if divide_by_mdd:
            max_dd = abs(stats.max_drawdown(series))
            values[name] = cagr_value / max_dd if max_dd else math.nan
        else:
            values[name] = cagr_value * multiplier
    return values


def _cagr(series: pl.Series, *, rf: float, compounded: bool, periods: int) -> float:
    values = [float(value) - rf for value in series.drop_nulls().to_list()]
    if not values:
        return math.nan
    n_years = len(values) / periods
    total = math.prod(1 + value for value in values) if compounded else sum(values) + 1
    return total ** (1 / n_years) - 1


def _period_dates(
    frame: pl.DataFrame,
    names: list[str],
    dates: list[Any] | None,
    *,
    first: bool,
) -> dict[str, Any]:
    if dates is None:
        return dict.fromkeys(names)
    result: dict[str, Any] = {}
    for name in names:
        values = frame[name].to_list()
        indexes = [idx for idx, value in enumerate(values) if value is not None]
        result[name] = _date_at(dates, indexes[0 if first else -1]) if indexes else None
    return result


def _date_at(dates: list[Any] | None, idx: int) -> Any:
    if dates is None:
        return None
    return dates[idx]


def _frame_value(result: Any, column: str, multiplier: float = 1.0) -> Any:
    if isinstance(result, pl.DataFrame):
        return _number(result[column][0], multiplier=multiplier)
    return _number(result, multiplier=multiplier)


def _greek(series: pl.Series, benchmark: pl.Series, field: str) -> float:
    greeks = stats.greeks(series, benchmark)
    return float(cast(dict[str, float], greeks[series.name][0])[field])


def _number(value: Any, multiplier: float = 1.0) -> Any:
    if isinstance(value, dict):
        return value
    if isinstance(value, int):
        return value * multiplier
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return value
    return numeric * multiplier


def _sum(series: pl.Series) -> float:
    return sum(float(value) for value in series.drop_nulls().to_list())


def _format_date(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return str(value)


def _format_value(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return "-"
        rounded = round(value, 2)
        if rounded == 0:
            rounded = 0
        return str(rounded)
    return str(value)


def _build_output(rows: list[tuple[str, dict[str, Any]]], output_names: list[str]) -> pl.DataFrame:
    data: dict[str, list[str]] = {"Metric": []}
    for name in output_names:
        data[name] = []
    for label, values in rows:
        data["Metric"].append(label)
        for name in output_names:
            data[name].append(_format_value(values.get(name)))
    return pl.DataFrame(data)
