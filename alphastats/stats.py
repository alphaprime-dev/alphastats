import math
from statistics import NormalDist
from typing import cast, overload

import polars as pl
import polars.selectors as cs

from alphastats._utils import (
    BENCHMARK_RETURNS_COLNAME,
    RETURNS_COLUMNS_SELECTOR,
    get_temporal_column,
    prepare_benchmark,
    to_excess_returns,
    to_lazy,
)
from alphastats.exceptions import NoTemporalColumnError

_NORMAL = NormalDist()


@overload
def comp(returns: pl.Series) -> float: ...


@overload
def comp(returns: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame: ...


def comp(returns: pl.Series | pl.DataFrame | pl.LazyFrame) -> float | pl.DataFrame:
    """
    Compute total compounded returns.

    Args:
        returns: Returns series or dataframe

    Returns:
        Compounded return(s)
    """
    returns_ldf = to_lazy(returns)

    res = returns_ldf.select(_comp(RETURNS_COLUMNS_SELECTOR)).collect()

    if isinstance(returns, pl.Series):
        return res.item()
    else:
        return res


def _comp(expr: pl.Expr) -> pl.Expr:
    return (expr + 1).product() - 1


def cagr(
    returns: pl.DataFrame | pl.LazyFrame,
    rf: float | None = None,
    compound: bool = True,
    periods: int = 252,
) -> pl.DataFrame:
    """
    Calculates the Compound Annual Growth Rate
    (CAGR%) of access returns

    If rf is non-zero, you must specify periods.
    In this case, rf is assumed to be expressed in yearly (annualized) terms

    Args:
        returns: Returns dataframe or lazyframe
        rf: Risk-free rate
        compound: Whether to compound the returns
        periods: Number of periods in a year

    Returns:
        CAGR%
    """
    returns_ldf = to_lazy(returns)

    excess_returns = to_excess_returns(RETURNS_COLUMNS_SELECTOR, rf)
    temporal_col = get_temporal_column(returns_ldf)
    if temporal_col is None:
        raise NoTemporalColumnError

    n_years = RETURNS_COLUMNS_SELECTOR.count() / periods

    if compound:
        expr = _comp(excess_returns).add(1).pow(1 / n_years).sub(1)
    else:
        expr = (excess_returns).sum().add(1).pow(1 / n_years).sub(1)

    return returns_ldf.select(expr).collect()


@overload
def max_drawdown(returns: pl.Series) -> float: ...


@overload
def max_drawdown(returns: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame: ...


def max_drawdown(returns: pl.Series | pl.DataFrame | pl.LazyFrame) -> float | pl.DataFrame:
    """
    Calculates the maximum drawdown of returns.

    Maximum drawdown is the largest peak-to-trough decline in the cumulative returns.

    Args:
        returns: Returns series or dataframe

    Returns:
        Maximum drawdown value(s)
    """
    returns_ldf = to_lazy(returns)

    max_drawdown = _drawdowns(RETURNS_COLUMNS_SELECTOR).min()

    res = returns_ldf.select(max_drawdown).collect()

    if isinstance(returns, pl.Series):
        return res.item()
    else:
        return res


@overload
def sharpe(
    returns: pl.Series,
    rf: float | pl.Series | None = None,
    periods: int = 252,
    annualize: bool = True,
) -> float: ...


@overload
def sharpe(
    returns: pl.DataFrame | pl.LazyFrame,
    rf: float | pl.Series | None = None,
    periods: int = 252,
    annualize: bool = True,
) -> pl.DataFrame: ...


def sharpe(
    returns: pl.Series | pl.DataFrame | pl.LazyFrame,
    rf: float | pl.Series | None = None,
    periods: int = 252,
    annualize: bool = True,
) -> float | pl.DataFrame:
    """
    Calculates the Sharpe ratio of excess returns.

    The Sharpe ratio measures risk-adjusted return by dividing excess return by volatility.

    Args:
        returns: Returns series or dataframe
        rf: Risk-free rate expressed as yearly (annualized) return
        periods: Frequency of returns (252 for daily, 12 for monthly)
        annualize: Whether to annualize the Sharpe ratio

    Returns:
        Sharpe ratio value(s)
    """
    returns_ldf = to_lazy(returns)

    excess_returns = to_excess_returns(RETURNS_COLUMNS_SELECTOR, rf)
    sharpe = excess_returns.mean() / excess_returns.std(ddof=1)

    if annualize:
        sharpe = sharpe * (periods**0.5)

    res = returns_ldf.select(sharpe).collect()

    if isinstance(returns, pl.Series):
        return res.item()
    else:
        return res


@overload
def sortino(
    returns: pl.Series,
    rf: float | pl.Series | None = None,
    periods: int = 252,
    annualize: bool = True,
) -> float: ...


@overload
def sortino(
    returns: pl.DataFrame | pl.LazyFrame,
    rf: float | pl.Series | None = None,
    periods: int = 252,
    annualize: bool = True,
) -> pl.DataFrame: ...


def sortino(
    returns: pl.Series | pl.DataFrame | pl.LazyFrame,
    rf: float | pl.Series | None = None,
    periods: int = 252,
    annualize: bool = True,
) -> float | pl.DataFrame:
    """
    Calculates the Sortino ratio of excess returns.

    The Sortino ratio measures risk-adjusted return by dividing excess return by downside risk.

    Args:
        returns: Returns series or dataframe
        rf: Risk-free rate per period (or per-period series)
        periods: Frequency of returns (252 for daily, 12 for monthly)
        annualize: Whether to annualize the Sortino ratio

    Returns:
        Sortino ratio value(s)
    """
    returns_ldf = to_lazy(returns)

    excess_returns = to_excess_returns(RETURNS_COLUMNS_SELECTOR, rf)
    downside = pl.when(excess_returns < 0).then(excess_returns).otherwise(0)
    downside_risk = downside.pow(2).mean().sqrt()

    sortino_expr = excess_returns.mean() / downside_risk

    if annualize:
        sortino_expr = sortino_expr * (periods**0.5)

    res = returns_ldf.select(sortino_expr).collect()

    if isinstance(returns, pl.Series):
        return res.item()
    else:
        return res


@overload
def probabilistic_sharpe_ratio(
    returns: pl.Series,
    rf: float = 0.0,
    periods: int = 252,
    annualize: bool = False,
    smart: bool = False,
) -> float: ...


@overload
def probabilistic_sharpe_ratio(
    returns: pl.DataFrame | pl.LazyFrame,
    rf: float = 0.0,
    periods: int = 252,
    annualize: bool = False,
    smart: bool = False,
) -> pl.DataFrame: ...


def probabilistic_sharpe_ratio(
    returns: pl.Series | pl.DataFrame | pl.LazyFrame,
    rf: float = 0.0,
    periods: int = 252,
    annualize: bool = False,
    smart: bool = False,
) -> float | pl.DataFrame:
    """
    Compute the QuantStats-compatible Probabilistic Sharpe Ratio (PSR).

    Args:
        returns: Returns series or dataframe
        rf: Risk-free rate subtracted from the Sharpe ratio
        periods: Number of periods in a year
        annualize: Whether to annualize the probability value
        smart: Whether to apply autocorrelation penalty to the base Sharpe ratio

    Returns:
        Probability value(s) in [0, 1]
    """
    returns_ldf = to_lazy(returns)
    base = (
        smart_sharpe(returns, periods=periods, annualize=False)
        if smart
        else sharpe(returns, periods=periods, annualize=False)
    )
    skewness = skew(returns)
    excess_kurtosis = kurtosis(returns)
    counts = returns_ldf.select(RETURNS_COLUMNS_SELECTOR.count()).collect()

    def _probability(base_value: float, skew_value: float, kurtosis_value: float, n: int) -> float:
        sigma = math.sqrt(
            (
                1
                + (0.5 * base_value**2)
                - (skew_value * base_value)
                + (((kurtosis_value - 3) / 4) * base_value**2)
            )
            / (n - 1)
        )
        probability = _NORMAL.cdf((base_value - rf) / sigma)
        if annualize:
            return probability * (252**0.5)
        return probability

    if isinstance(returns, pl.Series):
        return _probability(
            float(cast(float, base)),
            float(cast(float, skewness)),
            float(cast(float, excess_kurtosis)),
            counts.item(),
        )

    base_df = cast(pl.DataFrame, base)
    skew_df = cast(pl.DataFrame, skewness)
    kurtosis_df = cast(pl.DataFrame, excess_kurtosis)
    return pl.DataFrame(
        {
            col_name: [
                _probability(
                    base_df[col_name][0],
                    skew_df[col_name][0],
                    kurtosis_df[col_name][0],
                    counts[col_name][0],
                )
            ]
            for col_name in base_df.columns
        }
    )


def psr(
    returns: pl.Series | pl.DataFrame | pl.LazyFrame,
    rf: float = 0.0,
    periods: int = 252,
    annualize: bool = False,
    smart: bool = False,
) -> float | pl.DataFrame:
    """Alias for probabilistic_sharpe_ratio."""
    return probabilistic_sharpe_ratio(
        returns, rf=rf, periods=periods, annualize=annualize, smart=smart
    )


@overload
def volatility(returns: pl.Series, periods: int = 252, annualize: bool = True) -> float: ...


@overload
def volatility(
    returns: pl.DataFrame | pl.LazyFrame, periods: int = 252, annualize: bool = True
) -> pl.DataFrame: ...


def volatility(
    returns: pl.Series | pl.DataFrame | pl.LazyFrame,
    periods: int = 252,
    annualize: bool = True,
) -> float | pl.DataFrame:
    """
    Calculates the volatility (standard deviation) of returns.

    Args:
        returns: Returns series or dataframe
        periods: Frequency of returns (252 for daily, 12 for monthly)
        annualize: Whether to annualize the volatility

    Returns:
        Volatility value(s)
    """
    returns_ldf = to_lazy(returns)

    std_expr = RETURNS_COLUMNS_SELECTOR.std(ddof=1)

    if annualize:
        std_expr = std_expr * (periods**0.5)

    res = returns_ldf.select(std_expr).collect()

    if isinstance(returns, pl.Series):
        return res.item()
    else:
        return res


@overload
def to_drawdowns(returns: pl.Series) -> pl.Series: ...


@overload
def to_drawdowns(returns: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame: ...


def to_drawdowns(
    returns: pl.Series | pl.DataFrame | pl.LazyFrame,
) -> pl.Series | pl.DataFrame:
    """
    Convert returns to drawdowns.

    Drawdowns shows the percentage decline from the previous peak at each point in time.

    Args:
        returns: Returns series or dataframe

    Returns:
        Drawdowns with same structure as input
    """
    returns_ldf = to_lazy(returns)

    res = returns_ldf.with_columns(_drawdowns(RETURNS_COLUMNS_SELECTOR)).collect()

    if isinstance(returns, pl.Series):
        return res.to_series()
    else:
        return res


def _drawdowns(expr: pl.Expr) -> pl.Expr:
    wealth_index = (expr + 1).cum_prod()
    running_max_wealth_index = wealth_index.cum_max()
    drawdowns = (wealth_index / running_max_wealth_index) - 1
    return drawdowns.clip(lower_bound=None, upper_bound=0)


def greeks(
    returns: pl.Series | pl.DataFrame | pl.LazyFrame,
    benchmark: pl.Series | pl.DataFrame | pl.LazyFrame,
    periods: int = 252,
) -> pl.DataFrame:
    """
    Calculate CAPM alpha & beta for every numeric column in `returns`
    and return them as struct columns (alpha, beta).

    Returns:
        1-row DataFrame with struct columns (α·β) for each numeric column

    Example:
        ┌─────────────┬──────────────┬──────┬──────────────┐
        │ col_1       │ col_2        │ ...  │ col_n        │
        │ ---         │ ---          │ ---  │ ---          │
        │ struct[α·β] │ struct[α·β]  │ ...  │ struct[α·β]  │
        └─────────────┴──────────────┴──────┴──────────────┘
    """
    returns_ldf = to_lazy(returns)
    benchmark_ldf = prepare_benchmark(to_lazy(benchmark))

    returns_temporal_col = get_temporal_column(returns_ldf)
    benchmark_temporal_col = get_temporal_column(benchmark_ldf)

    if returns_temporal_col is not None and benchmark_temporal_col is not None:
        joined = returns_ldf.join_asof(
            benchmark_ldf,
            left_on=returns_temporal_col,
            right_on=benchmark_temporal_col,
        )
    else:
        joined = pl.concat([returns_ldf, benchmark_ldf], how="horizontal")

    strategy_returns_cols = RETURNS_COLUMNS_SELECTOR - cs.by_name(BENCHMARK_RETURNS_COLNAME)

    exprs: list[pl.Expr] = []
    for col_name in cs.expand_selector(joined, strategy_returns_cols):
        beta = pl.cov(col_name, BENCHMARK_RETURNS_COLNAME, ddof=1) / pl.var(
            BENCHMARK_RETURNS_COLNAME, ddof=1
        )
        alpha = pl.mean(col_name) - beta * pl.mean(BENCHMARK_RETURNS_COLNAME)

        exprs.append(
            pl.struct(
                [
                    (alpha * periods).alias("alpha"),
                    beta.alias("beta"),
                ]
            ).alias(col_name)
        )

    return joined.select(exprs).collect()


def calmar(returns: pl.DataFrame | pl.LazyFrame, periods: int = 252) -> pl.DataFrame:
    """
    Calculate the Calmar ratio for each numeric return column.

    Calmar = CAGR / |Max Drawdown|

    Notes:
        - Requires a temporal column to compute CAGR, similar to `cagr`.
        - Uses compounded CAGR and the absolute value of maximum drawdown over
          the entire period.

    Args:
        returns: Returns dataframe or lazyframe (must include a temporal column)
        periods: Number of periods in a year (e.g., 252 for daily, 12 for monthly)

    Returns:
        1-row DataFrame with Calmar ratio per numeric column
    """
    returns_ldf = to_lazy(returns)

    temporal_col = get_temporal_column(returns_ldf)
    if temporal_col is None:
        raise NoTemporalColumnError

    n_years = RETURNS_COLUMNS_SELECTOR.count() / periods

    cagr_expr = _comp(RETURNS_COLUMNS_SELECTOR).add(1).pow(1 / n_years).sub(1)

    max_dd_abs_expr = _drawdowns(RETURNS_COLUMNS_SELECTOR).min().abs()

    calmar_expr = cagr_expr / max_dd_abs_expr

    return returns_ldf.select(calmar_expr).collect()


@overload
def information_ratio(
    returns: pl.Series,
    benchmark: pl.Series | pl.DataFrame | pl.LazyFrame,
    periods: int = 252,
    annualize: bool = False,
) -> float: ...


@overload
def information_ratio(
    returns: pl.DataFrame | pl.LazyFrame,
    benchmark: pl.Series | pl.DataFrame | pl.LazyFrame,
    periods: int = 252,
    annualize: bool = False,
) -> pl.DataFrame: ...


def information_ratio(
    returns: pl.Series | pl.DataFrame | pl.LazyFrame,
    benchmark: pl.Series | pl.DataFrame | pl.LazyFrame,
    periods: int = 252,
    annualize: bool = False,
) -> float | pl.DataFrame:
    """
    Compute the Information Ratio for each numeric return column.

    Information Ratio = mean(active returns) / std(active returns)
    where active returns = strategy - benchmark. When annualize=True,
    the ratio is scaled by sqrt(periods).

    Args:
        returns: Strategy returns series/dataframe
        benchmark: Benchmark returns (series/dataframe/lazyframe)
        periods: Number of periods in a year
        annualize: Whether to annualize the ratio

    Returns:
        Information Ratio value(s)
    """
    returns_ldf = to_lazy(returns)
    benchmark_ldf = prepare_benchmark(to_lazy(benchmark))

    returns_temporal_col = get_temporal_column(returns_ldf)
    benchmark_temporal_col = get_temporal_column(benchmark_ldf)

    if returns_temporal_col is not None and benchmark_temporal_col is not None:
        joined = returns_ldf.join_asof(
            benchmark_ldf,
            left_on=returns_temporal_col,
            right_on=benchmark_temporal_col,
        )
    else:
        joined = pl.concat([returns_ldf, benchmark_ldf], how="horizontal")

    strategy_returns_cols = RETURNS_COLUMNS_SELECTOR - cs.by_name(BENCHMARK_RETURNS_COLNAME)

    exprs: list[pl.Expr] = []
    for col_name in cs.expand_selector(joined, strategy_returns_cols):
        active = pl.col(col_name) - pl.col(BENCHMARK_RETURNS_COLNAME)
        ir = active.mean() / active.std(ddof=1)
        if annualize:
            ir = ir * (periods**0.5)
        exprs.append(ir.alias(col_name))

    res = joined.select(exprs).collect()

    if isinstance(returns, pl.Series):
        return res.item()
    else:
        return res


@overload
def cpc_index(returns: pl.Series) -> float: ...


@overload
def cpc_index(returns: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame: ...


def cpc_index(returns: pl.Series | pl.DataFrame | pl.LazyFrame) -> float | pl.DataFrame:
    """
    Compute the CPC Index for each numeric return column.

    CPC Index = Profit Factor * Payoff Ratio * Win Rate

    Where:
        - Profit Factor = sum of gains / absolute sum of losses
        - Payoff Ratio = average gain / average loss (absolute)
        - Win Rate = number of gain periods / total non-null periods

    Args:
        returns: Returns series or dataframe

    Returns:
        CPC Index value(s)
    """
    returns_ldf = to_lazy(returns)

    r = RETURNS_COLUMNS_SELECTOR

    gains_sum = pl.when(r >= 0).then(r).otherwise(0).sum()
    losses_sum_abs = pl.when(r < 0).then(r).otherwise(0).sum().abs()

    wins_count = (r > 0).cast(pl.Int64).sum()
    losses_count = (r < 0).cast(pl.Int64).sum()
    non_zero_count = (r != 0).cast(pl.Int64).sum()

    avg_win = gains_sum / wins_count
    avg_loss_abs = losses_sum_abs / losses_count

    profit_factor = gains_sum / losses_sum_abs
    payoff_ratio = avg_win / avg_loss_abs
    win_rate = wins_count / non_zero_count

    expr = profit_factor * payoff_ratio * win_rate

    res = returns_ldf.select(expr).collect()

    if isinstance(returns, pl.Series):
        return res.item()
    else:
        return res


@overload
def exposure(returns: pl.Series) -> float: ...


@overload
def exposure(returns: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame: ...


def exposure(returns: pl.Series | pl.DataFrame | pl.LazyFrame) -> float | pl.DataFrame:
    """
    Calculate Time in Market (exposure) for each numeric return column.

    Exposure is the fraction of periods with non-zero returns over total
    non-null periods. Returned as a decimal between 0 and 1.

    Args:
        returns: Returns series or dataframe

    Returns:
        Exposure value(s)
    """
    returns_ldf = to_lazy(returns)

    r = RETURNS_COLUMNS_SELECTOR
    non_zero = r.ne(0).cast(pl.Int64).sum()
    total = r.is_not_null().cast(pl.Int64).sum()
    expr = non_zero / total

    res = returns_ldf.select(expr).collect()

    if isinstance(returns, pl.Series):
        return res.item()
    else:
        return res


@overload
def omega(returns: pl.Series, threshold: float = 0.0) -> float: ...


@overload
def omega(returns: pl.DataFrame | pl.LazyFrame, threshold: float = 0.0) -> pl.DataFrame: ...


def omega(
    returns: pl.Series | pl.DataFrame | pl.LazyFrame, threshold: float = 0.0
) -> float | pl.DataFrame:
    """
    Compute the Omega ratio for each numeric return column at a given threshold.

    Discrete-sample approximation:
        Omega(θ) = sum(max(0, r - θ)) / sum(max(0, θ - r))

    Args:
        returns: Returns series or dataframe
        threshold: Target return θ per period (default 0.0)

    Returns:
        Omega ratio value(s)
    """
    returns_ldf = to_lazy(returns)

    r = RETURNS_COLUMNS_SELECTOR
    t = pl.lit(float(threshold))

    diff = r - t
    gains = diff.clip(lower_bound=0).sum()
    losses = (-diff).clip(lower_bound=0).sum()

    expr = gains / losses

    res = returns_ldf.select(expr).collect()

    if isinstance(returns, pl.Series):
        return res.item()
    else:
        return res


def _metric_result(
    returns: pl.Series | pl.DataFrame | pl.LazyFrame, returns_ldf: pl.LazyFrame, expr: pl.Expr
) -> float | pl.DataFrame:
    res = returns_ldf.select(expr).collect()
    if isinstance(returns, pl.Series):
        return res.item()
    return res


def _numeric_column_names(ldf: pl.LazyFrame) -> list[str]:
    return list(cs.expand_selector(ldf, RETURNS_COLUMNS_SELECTOR))


def _temporal_column_name(ldf: pl.LazyFrame) -> str:
    column_names = cs.expand_selector(ldf, cs.temporal())
    if len(column_names) != 1:
        temporal_col = get_temporal_column(ldf)
        if temporal_col is None:
            raise NoTemporalColumnError
    return column_names[0]


def _simple_returns(
    returns_ldf: pl.LazyFrame, aggregate: str | None, compounded: bool
) -> pl.LazyFrame:
    if aggregate is None or "day" in aggregate.lower():
        return returns_ldf.select(RETURNS_COLUMNS_SELECTOR)

    temporal_col_name = _temporal_column_name(returns_ldf)
    temporal_col = pl.col(temporal_col_name)
    aggregate_key = aggregate.lower()
    return_col_names = _numeric_column_names(returns_ldf)

    group_exprs: list[pl.Expr]
    if "week" in aggregate_key or aggregate_key in {"w", "eow"}:
        group_exprs = [temporal_col.dt.year().alias("_year"), temporal_col.dt.week().alias("_week")]
    elif "month" in aggregate_key or aggregate_key in {"m", "me", "eom"}:
        group_exprs = [
            temporal_col.dt.year().alias("_year"),
            temporal_col.dt.month().alias("_month"),
        ]
    elif "quarter" in aggregate_key or aggregate_key in {"q", "qe", "eoq"}:
        group_exprs = [
            temporal_col.dt.year().alias("_year"),
            temporal_col.dt.quarter().alias("_quarter"),
        ]
    elif aggregate_key in {"y", "ye", "eoy", "year", "yearly", "annual"}:
        group_exprs = [temporal_col.dt.year().alias("_year")]
    else:
        return returns_ldf.select(RETURNS_COLUMNS_SELECTOR)

    exprs = [
        (_comp(pl.col(col_name)) if compounded else pl.col(col_name).sum()).alias(col_name)
        for col_name in return_col_names
    ]
    return (
        returns_ldf.with_columns(group_exprs)
        .group_by([expr.meta.output_name() for expr in group_exprs], maintain_order=True)
        .agg(exprs)
        .select(return_col_names)
    )


def _aggregate_metric(
    returns: pl.Series | pl.DataFrame | pl.LazyFrame,
    aggregate: str | None,
    compounded: bool,
    expr: pl.Expr,
) -> float | pl.DataFrame:
    returns_ldf = _simple_returns(to_lazy(returns), aggregate, compounded)
    return _metric_result(returns, returns_ldf, expr)


@overload
def expected_return(
    returns: pl.Series, aggregate: str | None = None, compounded: bool = True
) -> float: ...


@overload
def expected_return(
    returns: pl.DataFrame | pl.LazyFrame, aggregate: str | None = None, compounded: bool = True
) -> pl.DataFrame: ...


def expected_return(
    returns: pl.Series | pl.DataFrame | pl.LazyFrame,
    aggregate: str | None = None,
    compounded: bool = True,
) -> float | pl.DataFrame:
    """Calculate geometric expected return for each numeric return column."""
    ldf = _simple_returns(to_lazy(returns), aggregate, compounded)
    expr = (RETURNS_COLUMNS_SELECTOR + 1).product().pow(1 / RETURNS_COLUMNS_SELECTOR.count()).sub(1)
    return _metric_result(returns, ldf, expr)


geometric_mean = expected_return


@overload
def best(returns: pl.Series, aggregate: str | None = None, compounded: bool = True) -> float: ...


@overload
def best(
    returns: pl.DataFrame | pl.LazyFrame, aggregate: str | None = None, compounded: bool = True
) -> pl.DataFrame: ...


def best(
    returns: pl.Series | pl.DataFrame | pl.LazyFrame,
    aggregate: str | None = None,
    compounded: bool = True,
) -> float | pl.DataFrame:
    """Return the best return over the requested aggregation period."""
    return _aggregate_metric(returns, aggregate, compounded, RETURNS_COLUMNS_SELECTOR.max())


@overload
def worst(returns: pl.Series, aggregate: str | None = None, compounded: bool = True) -> float: ...


@overload
def worst(
    returns: pl.DataFrame | pl.LazyFrame, aggregate: str | None = None, compounded: bool = True
) -> pl.DataFrame: ...


def worst(
    returns: pl.Series | pl.DataFrame | pl.LazyFrame,
    aggregate: str | None = None,
    compounded: bool = True,
) -> float | pl.DataFrame:
    """Return the worst return over the requested aggregation period."""
    return _aggregate_metric(returns, aggregate, compounded, RETURNS_COLUMNS_SELECTOR.min())


@overload
def skew(returns: pl.Series) -> float: ...


@overload
def skew(returns: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame: ...


def skew(returns: pl.Series | pl.DataFrame | pl.LazyFrame) -> float | pl.DataFrame:
    """Calculate returns skewness."""
    returns_ldf = to_lazy(returns)
    r = RETURNS_COLUMNS_SELECTOR
    n = r.count()
    mean = r.mean()
    centered = r - mean
    m2 = centered.pow(2).mean()
    m3 = centered.pow(3).mean()
    expr = (n * (n - 1)).sqrt() / (n - 2) * m3 / m2.pow(1.5)
    return _metric_result(returns, returns_ldf, expr)


@overload
def kurtosis(returns: pl.Series) -> float: ...


@overload
def kurtosis(returns: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame: ...


def kurtosis(returns: pl.Series | pl.DataFrame | pl.LazyFrame) -> float | pl.DataFrame:
    """Calculate returns excess kurtosis."""
    returns_ldf = to_lazy(returns)
    r = RETURNS_COLUMNS_SELECTOR
    n = r.count()
    mean = r.mean()
    centered = r - mean
    m2 = centered.pow(2).mean()
    m4 = centered.pow(4).mean()
    g2 = m4 / m2.pow(2) - 3
    expr = ((n - 1) / ((n - 2) * (n - 3))) * ((n + 1) * g2 + 6)
    return _metric_result(returns, returns_ldf, expr)


@overload
def avg_return(
    returns: pl.Series, aggregate: str | None = None, compounded: bool = True
) -> float: ...


@overload
def avg_return(
    returns: pl.DataFrame | pl.LazyFrame, aggregate: str | None = None, compounded: bool = True
) -> pl.DataFrame: ...


def avg_return(
    returns: pl.Series | pl.DataFrame | pl.LazyFrame,
    aggregate: str | None = None,
    compounded: bool = True,
) -> float | pl.DataFrame:
    """Calculate the mean of non-zero returns."""
    ldf = _simple_returns(to_lazy(returns), aggregate, compounded)
    r = RETURNS_COLUMNS_SELECTOR
    return _metric_result(returns, ldf, pl.when(r != 0).then(r).otherwise(None).mean())


@overload
def avg_win(returns: pl.Series, aggregate: str | None = None, compounded: bool = True) -> float: ...


@overload
def avg_win(
    returns: pl.DataFrame | pl.LazyFrame, aggregate: str | None = None, compounded: bool = True
) -> pl.DataFrame: ...


def avg_win(
    returns: pl.Series | pl.DataFrame | pl.LazyFrame,
    aggregate: str | None = None,
    compounded: bool = True,
) -> float | pl.DataFrame:
    """Calculate the mean of positive returns."""
    ldf = _simple_returns(to_lazy(returns), aggregate, compounded)
    r = RETURNS_COLUMNS_SELECTOR
    return _metric_result(returns, ldf, pl.when(r > 0).then(r).otherwise(None).mean())


@overload
def avg_loss(
    returns: pl.Series, aggregate: str | None = None, compounded: bool = True
) -> float: ...


@overload
def avg_loss(
    returns: pl.DataFrame | pl.LazyFrame, aggregate: str | None = None, compounded: bool = True
) -> pl.DataFrame: ...


def avg_loss(
    returns: pl.Series | pl.DataFrame | pl.LazyFrame,
    aggregate: str | None = None,
    compounded: bool = True,
) -> float | pl.DataFrame:
    """Calculate the mean of negative returns."""
    ldf = _simple_returns(to_lazy(returns), aggregate, compounded)
    r = RETURNS_COLUMNS_SELECTOR
    return _metric_result(returns, ldf, pl.when(r < 0).then(r).otherwise(None).mean())


@overload
def win_rate(
    returns: pl.Series, aggregate: str | None = None, compounded: bool = True
) -> float: ...


@overload
def win_rate(
    returns: pl.DataFrame | pl.LazyFrame, aggregate: str | None = None, compounded: bool = True
) -> pl.DataFrame: ...


def win_rate(
    returns: pl.Series | pl.DataFrame | pl.LazyFrame,
    aggregate: str | None = None,
    compounded: bool = True,
) -> float | pl.DataFrame:
    """Calculate the ratio of positive returns to non-zero returns."""
    ldf = _simple_returns(to_lazy(returns), aggregate, compounded)
    r = RETURNS_COLUMNS_SELECTOR
    wins = (r > 0).cast(pl.Int64).sum()
    non_zero = (r != 0).cast(pl.Int64).sum()
    return _metric_result(returns, ldf, wins / non_zero)


@overload
def payoff_ratio(returns: pl.Series) -> float: ...


@overload
def payoff_ratio(returns: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame: ...


def payoff_ratio(returns: pl.Series | pl.DataFrame | pl.LazyFrame) -> float | pl.DataFrame:
    """Calculate average win divided by absolute average loss."""
    returns_ldf = to_lazy(returns)
    r = RETURNS_COLUMNS_SELECTOR
    avg_win_expr = pl.when(r > 0).then(r).otherwise(None).mean()
    avg_loss_expr = pl.when(r < 0).then(r).otherwise(None).mean().abs()
    return _metric_result(returns, returns_ldf, avg_win_expr / avg_loss_expr)


@overload
def profit_factor(returns: pl.Series) -> float: ...


@overload
def profit_factor(returns: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame: ...


def profit_factor(returns: pl.Series | pl.DataFrame | pl.LazyFrame) -> float | pl.DataFrame:
    """Calculate total wins divided by absolute total losses."""
    returns_ldf = to_lazy(returns)
    r = RETURNS_COLUMNS_SELECTOR
    wins = pl.when(r >= 0).then(r).otherwise(0).sum()
    losses = pl.when(r < 0).then(r).otherwise(0).sum().abs()
    return _metric_result(returns, returns_ldf, wins / losses)


@overload
def gain_to_pain_ratio(returns: pl.Series, rf: float = 0.0) -> float: ...


@overload
def gain_to_pain_ratio(returns: pl.DataFrame | pl.LazyFrame, rf: float = 0.0) -> pl.DataFrame: ...


def gain_to_pain_ratio(
    returns: pl.Series | pl.DataFrame | pl.LazyFrame, rf: float = 0.0
) -> float | pl.DataFrame:
    """Calculate total returns divided by absolute negative returns."""
    returns_ldf = to_lazy(returns)
    r = to_excess_returns(RETURNS_COLUMNS_SELECTOR, rf)
    downside = pl.when(r < 0).then(r).otherwise(0).sum().abs()
    return _metric_result(returns, returns_ldf, r.sum() / downside)


def gain_to_pain_ratio_1m(
    returns: pl.Series | pl.DataFrame | pl.LazyFrame, rf: float = 0.0
) -> float | pl.DataFrame:
    """Calculate gain-to-pain ratio on summed monthly returns."""
    ldf = _simple_returns(to_lazy(returns), "month", compounded=False)
    r = to_excess_returns(RETURNS_COLUMNS_SELECTOR, rf)
    downside = pl.when(r < 0).then(r).otherwise(0).sum().abs()
    return _metric_result(returns, ldf, r.sum() / downside)


gain_to_pain_ratio_monthly = gain_to_pain_ratio_1m


@overload
def common_sense_ratio(returns: pl.Series) -> float: ...


@overload
def common_sense_ratio(returns: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame: ...


def common_sense_ratio(
    returns: pl.Series | pl.DataFrame | pl.LazyFrame,
) -> float | pl.DataFrame:
    """Calculate profit factor multiplied by tail ratio."""
    returns_ldf = to_lazy(returns)
    r = RETURNS_COLUMNS_SELECTOR
    wins = pl.when(r >= 0).then(r).otherwise(0).sum()
    losses = pl.when(r < 0).then(r).otherwise(0).sum().abs()
    tail = (
        r.quantile(0.95, interpolation="linear").abs()
        / r.quantile(0.05, interpolation="linear").abs()
    )
    return _metric_result(returns, returns_ldf, (wins / losses) * tail)


@overload
def tail_ratio(returns: pl.Series, cutoff: float = 0.95) -> float: ...


@overload
def tail_ratio(returns: pl.DataFrame | pl.LazyFrame, cutoff: float = 0.95) -> pl.DataFrame: ...


def tail_ratio(
    returns: pl.Series | pl.DataFrame | pl.LazyFrame, cutoff: float = 0.95
) -> float | pl.DataFrame:
    """Calculate absolute upper-tail quantile divided by lower-tail quantile."""
    returns_ldf = to_lazy(returns)
    r = RETURNS_COLUMNS_SELECTOR
    expr = (
        r.quantile(cutoff, interpolation="linear").abs()
        / r.quantile(1 - cutoff, interpolation="linear").abs()
    )
    return _metric_result(returns, returns_ldf, expr)


@overload
def outlier_win_ratio(returns: pl.Series, quantile: float = 0.99) -> float: ...


@overload
def outlier_win_ratio(
    returns: pl.DataFrame | pl.LazyFrame, quantile: float = 0.99
) -> pl.DataFrame: ...


def outlier_win_ratio(
    returns: pl.Series | pl.DataFrame | pl.LazyFrame, quantile: float = 0.99
) -> float | pl.DataFrame:
    """Calculate high quantile divided by mean non-negative return."""
    returns_ldf = to_lazy(returns)
    r = RETURNS_COLUMNS_SELECTOR
    positive_mean = pl.when(r >= 0).then(r).otherwise(None).mean()
    return _metric_result(
        returns, returns_ldf, r.quantile(quantile, interpolation="linear") / positive_mean
    )


@overload
def outlier_loss_ratio(returns: pl.Series, quantile: float = 0.01) -> float: ...


@overload
def outlier_loss_ratio(
    returns: pl.DataFrame | pl.LazyFrame, quantile: float = 0.01
) -> pl.DataFrame: ...


def outlier_loss_ratio(
    returns: pl.Series | pl.DataFrame | pl.LazyFrame, quantile: float = 0.01
) -> float | pl.DataFrame:
    """Calculate low quantile divided by mean negative return."""
    returns_ldf = to_lazy(returns)
    r = RETURNS_COLUMNS_SELECTOR
    negative_mean = pl.when(r < 0).then(r).otherwise(None).mean()
    return _metric_result(
        returns, returns_ldf, r.quantile(quantile, interpolation="linear") / negative_mean
    )


@overload
def kelly_criterion(returns: pl.Series) -> float: ...


@overload
def kelly_criterion(returns: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame: ...


def kelly_criterion(returns: pl.Series | pl.DataFrame | pl.LazyFrame) -> float | pl.DataFrame:
    """Calculate Kelly Criterion allocation fraction."""
    returns_ldf = to_lazy(returns)
    r = RETURNS_COLUMNS_SELECTOR
    avg_win_expr = pl.when(r > 0).then(r).otherwise(None).mean()
    avg_loss_expr = pl.when(r < 0).then(r).otherwise(None).mean().abs()
    win_loss = avg_win_expr / avg_loss_expr
    wins = (r > 0).cast(pl.Int64).sum()
    non_zero = (r != 0).cast(pl.Int64).sum()
    win_prob = wins / non_zero
    lose_prob = win_prob.mul(-1).add(1)
    return _metric_result(returns, returns_ldf, ((win_loss * win_prob) - lose_prob) / win_loss)


@overload
def risk_of_ruin(returns: pl.Series) -> float: ...


@overload
def risk_of_ruin(returns: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame: ...


def risk_of_ruin(returns: pl.Series | pl.DataFrame | pl.LazyFrame) -> float | pl.DataFrame:
    """Calculate QuantStats-style risk of ruin from win rate and sample count."""
    returns_ldf = to_lazy(returns)
    r = RETURNS_COLUMNS_SELECTOR
    wins = (r > 0).cast(pl.Int64).sum()
    non_zero = (r != 0).cast(pl.Int64).sum()
    win_prob = wins / non_zero
    return _metric_result(
        returns, returns_ldf, (win_prob.mul(-1).add(1) / win_prob.add(1)).pow(r.count())
    )


ror = risk_of_ruin


@overload
def value_at_risk(returns: pl.Series, sigma: float = 1.0, confidence: float = 0.95) -> float: ...


@overload
def value_at_risk(
    returns: pl.DataFrame | pl.LazyFrame, sigma: float = 1.0, confidence: float = 0.95
) -> pl.DataFrame: ...


def value_at_risk(
    returns: pl.Series | pl.DataFrame | pl.LazyFrame,
    sigma: float = 1.0,
    confidence: float = 0.95,
) -> float | pl.DataFrame:
    """Calculate variance-covariance Value at Risk."""
    if confidence > 1:
        confidence = confidence / 100
    z_score = _NORMAL.inv_cdf(1 - confidence)
    returns_ldf = to_lazy(returns)
    r = RETURNS_COLUMNS_SELECTOR
    return _metric_result(returns, returns_ldf, r.mean() + z_score * sigma * r.std(ddof=1))


var = value_at_risk


@overload
def conditional_value_at_risk(
    returns: pl.Series, sigma: float = 1.0, confidence: float = 0.95
) -> float: ...


@overload
def conditional_value_at_risk(
    returns: pl.DataFrame | pl.LazyFrame, sigma: float = 1.0, confidence: float = 0.95
) -> pl.DataFrame: ...


def conditional_value_at_risk(
    returns: pl.Series | pl.DataFrame | pl.LazyFrame,
    sigma: float = 1.0,
    confidence: float = 0.95,
) -> float | pl.DataFrame:
    """Calculate Conditional Value at Risk, also known as Expected Shortfall."""
    if confidence > 1:
        confidence = confidence / 100
    z_score = _NORMAL.inv_cdf(1 - confidence)
    returns_ldf = to_lazy(returns)
    exprs: list[pl.Expr] = []
    for col_name in _numeric_column_names(returns_ldf):
        r = pl.col(col_name)
        var_expr = r.mean() + z_score * sigma * r.std(ddof=1)
        exprs.append(r.filter(r < var_expr).mean().fill_null(var_expr).alias(col_name))

    res = returns_ldf.select(exprs).collect()
    if isinstance(returns, pl.Series):
        return res.item()
    return res


cvar = conditional_value_at_risk
expected_shortfall = conditional_value_at_risk


def _consecutive_count(values: list[object], winning: bool) -> int:
    longest = 0
    current = 0
    for value in values:
        if value is None:
            current = 0
            continue
        if (value > 0) if winning else (value < 0):  # type: ignore[operator]
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def consecutive_wins(
    returns: pl.Series | pl.DataFrame | pl.LazyFrame,
    aggregate: str | None = None,
    compounded: bool = True,
) -> int | pl.DataFrame:
    """Calculate the longest positive-return streak."""
    ldf = _simple_returns(to_lazy(returns), aggregate, compounded)
    df = ldf.select(RETURNS_COLUMNS_SELECTOR).collect()
    values = {col: [_consecutive_count(df[col].to_list(), winning=True)] for col in df.columns}
    if isinstance(returns, pl.Series):
        return values[df.columns[0]][0]
    return pl.DataFrame(values)


def consecutive_losses(
    returns: pl.Series | pl.DataFrame | pl.LazyFrame,
    aggregate: str | None = None,
    compounded: bool = True,
) -> int | pl.DataFrame:
    """Calculate the longest negative-return streak."""
    ldf = _simple_returns(to_lazy(returns), aggregate, compounded)
    df = ldf.select(RETURNS_COLUMNS_SELECTOR).collect()
    values = {col: [_consecutive_count(df[col].to_list(), winning=False)] for col in df.columns}
    if isinstance(returns, pl.Series):
        return values[df.columns[0]][0]
    return pl.DataFrame(values)


def _autocorr_penalty_values(returns: pl.Series | pl.DataFrame | pl.LazyFrame) -> dict[str, float]:
    returns_ldf = to_lazy(returns)
    df = returns_ldf.select(RETURNS_COLUMNS_SELECTOR).collect()
    penalties: dict[str, float] = {}
    for col_name in df.columns:
        vals = [float(v) for v in df[col_name].drop_nulls().to_list()]
        num = len(vals)
        if num < 2:
            penalties[col_name] = math.nan
            continue
        mean_left = sum(vals[:-1]) / (num - 1)
        mean_right = sum(vals[1:]) / (num - 1)
        left_var = sum((v - mean_left) ** 2 for v in vals[:-1])
        right_var = sum((v - mean_right) ** 2 for v in vals[1:])
        if left_var == 0 or right_var == 0:
            penalties[col_name] = math.nan
            continue
        covariance = sum(
            (a - mean_left) * (b - mean_right) for a, b in zip(vals[:-1], vals[1:], strict=True)
        )
        coef = abs(covariance / math.sqrt(left_var * right_var))
        corr = sum(((num - x) / num) * (coef**x) for x in range(1, num))
        penalties[col_name] = math.sqrt(1 + 2 * corr)
    return penalties


@overload
def autocorr_penalty(returns: pl.Series) -> float: ...


@overload
def autocorr_penalty(returns: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame: ...


def autocorr_penalty(returns: pl.Series | pl.DataFrame | pl.LazyFrame) -> float | pl.DataFrame:
    """Calculate the autocorrelation penalty used by smart ratios."""
    penalties = _autocorr_penalty_values(returns)
    if isinstance(returns, pl.Series):
        return next(iter(penalties.values()))
    return pl.DataFrame({col: [value] for col, value in penalties.items()})


def smart_sharpe(
    returns: pl.Series | pl.DataFrame | pl.LazyFrame,
    rf: float | pl.Series | None = None,
    periods: int = 252,
    annualize: bool = True,
) -> float | pl.DataFrame:
    """Calculate Sharpe ratio with QuantStats' autocorrelation penalty."""
    penalties = _autocorr_penalty_values(returns)
    if isinstance(returns, pl.Series):
        base = sharpe(returns, rf=rf, periods=periods, annualize=annualize)
        return base / next(iter(penalties.values()))
    base_df = sharpe(returns, rf=rf, periods=periods, annualize=annualize)
    return pl.DataFrame({col: [base_df[col][0] / penalties[col]] for col in base_df.columns})


def smart_sortino(
    returns: pl.Series | pl.DataFrame | pl.LazyFrame,
    rf: float | pl.Series | None = None,
    periods: int = 252,
    annualize: bool = True,
) -> float | pl.DataFrame:
    """Calculate Sortino ratio with QuantStats' autocorrelation penalty."""
    penalties = _autocorr_penalty_values(returns)
    if isinstance(returns, pl.Series):
        base = sortino(returns, rf=rf, periods=periods, annualize=annualize)
        return base / next(iter(penalties.values()))
    base_df = sortino(returns, rf=rf, periods=periods, annualize=annualize)
    return pl.DataFrame({col: [base_df[col][0] / penalties[col]] for col in base_df.columns})


def adjusted_sortino(
    returns: pl.Series | pl.DataFrame | pl.LazyFrame,
    rf: float | pl.Series | None = None,
    periods: int = 252,
    annualize: bool = True,
    smart: bool = False,
) -> float | pl.DataFrame:
    """Calculate Jack Schwager's Sortino divided by sqrt(2)."""
    value = (
        smart_sortino(returns, rf=rf, periods=periods, annualize=annualize)
        if smart
        else sortino(returns, rf=rf, periods=periods, annualize=annualize)
    )
    if isinstance(value, pl.DataFrame):
        return value.select(RETURNS_COLUMNS_SELECTOR / math.sqrt(2))
    return value / math.sqrt(2)


sortino_sqrt2 = adjusted_sortino


def smart_adjusted_sortino(
    returns: pl.Series | pl.DataFrame | pl.LazyFrame,
    rf: float | pl.Series | None = None,
    periods: int = 252,
    annualize: bool = True,
) -> float | pl.DataFrame:
    """Calculate smart adjusted Sortino."""
    return adjusted_sortino(returns, rf=rf, periods=periods, annualize=annualize, smart=True)


smart_sortino_sqrt2 = smart_adjusted_sortino


def _joined_with_benchmark(
    returns: pl.Series | pl.DataFrame | pl.LazyFrame,
    benchmark: pl.Series | pl.DataFrame | pl.LazyFrame,
) -> pl.LazyFrame:
    returns_ldf = to_lazy(returns)
    benchmark_ldf = prepare_benchmark(to_lazy(benchmark))

    returns_temporal_col = get_temporal_column(returns_ldf)
    benchmark_temporal_col = get_temporal_column(benchmark_ldf)

    if returns_temporal_col is not None and benchmark_temporal_col is not None:
        return returns_ldf.join_asof(
            benchmark_ldf,
            left_on=returns_temporal_col,
            right_on=benchmark_temporal_col,
        )
    return pl.concat([returns_ldf, benchmark_ldf], how="horizontal")


@overload
def correlation(
    returns: pl.Series, benchmark: pl.Series | pl.DataFrame | pl.LazyFrame
) -> float: ...


@overload
def correlation(
    returns: pl.DataFrame | pl.LazyFrame, benchmark: pl.Series | pl.DataFrame | pl.LazyFrame
) -> pl.DataFrame: ...


def correlation(
    returns: pl.Series | pl.DataFrame | pl.LazyFrame,
    benchmark: pl.Series | pl.DataFrame | pl.LazyFrame,
) -> float | pl.DataFrame:
    """Calculate Pearson correlation against benchmark returns."""
    joined = _joined_with_benchmark(returns, benchmark)
    strategy_returns_cols = RETURNS_COLUMNS_SELECTOR - cs.by_name(BENCHMARK_RETURNS_COLNAME)
    exprs = [
        pl.corr(pl.col(col_name), pl.col(BENCHMARK_RETURNS_COLNAME)).alias(col_name)
        for col_name in cs.expand_selector(joined, strategy_returns_cols)
    ]
    res = joined.select(exprs).collect()
    if isinstance(returns, pl.Series):
        return res.item()
    return res


@overload
def r_squared(returns: pl.Series, benchmark: pl.Series | pl.DataFrame | pl.LazyFrame) -> float: ...


@overload
def r_squared(
    returns: pl.DataFrame | pl.LazyFrame, benchmark: pl.Series | pl.DataFrame | pl.LazyFrame
) -> pl.DataFrame: ...


def r_squared(
    returns: pl.Series | pl.DataFrame | pl.LazyFrame,
    benchmark: pl.Series | pl.DataFrame | pl.LazyFrame,
) -> float | pl.DataFrame:
    """Calculate R-squared against benchmark returns."""
    corr = correlation(returns, benchmark)
    if isinstance(corr, pl.DataFrame):
        return corr.select(RETURNS_COLUMNS_SELECTOR.pow(2))
    return corr**2


r2 = r_squared


@overload
def treynor_ratio(
    returns: pl.Series,
    benchmark: pl.Series | pl.DataFrame | pl.LazyFrame,
    rf: float = 0.0,
) -> float: ...


@overload
def treynor_ratio(
    returns: pl.DataFrame | pl.LazyFrame,
    benchmark: pl.Series | pl.DataFrame | pl.LazyFrame,
    rf: float = 0.0,
) -> pl.DataFrame: ...


def treynor_ratio(
    returns: pl.Series | pl.DataFrame | pl.LazyFrame,
    benchmark: pl.Series | pl.DataFrame | pl.LazyFrame,
    rf: float = 0.0,
) -> float | pl.DataFrame:
    """Calculate compounded excess return divided by CAPM beta."""
    joined = _joined_with_benchmark(returns, benchmark)
    strategy_returns_cols = RETURNS_COLUMNS_SELECTOR - cs.by_name(BENCHMARK_RETURNS_COLNAME)
    exprs: list[pl.Expr] = []
    for col_name in cs.expand_selector(joined, strategy_returns_cols):
        beta = pl.cov(col_name, BENCHMARK_RETURNS_COLNAME, ddof=1) / pl.var(
            BENCHMARK_RETURNS_COLNAME, ddof=1
        )
        exprs.append(((_comp(pl.col(col_name)) - rf) / beta).alias(col_name))
    res = joined.select(exprs).collect()
    if isinstance(returns, pl.Series):
        return res.item()
    return res


@overload
def recovery_factor(returns: pl.Series, rf: float = 0.0) -> float: ...


@overload
def recovery_factor(returns: pl.DataFrame | pl.LazyFrame, rf: float = 0.0) -> pl.DataFrame: ...


def recovery_factor(
    returns: pl.Series | pl.DataFrame | pl.LazyFrame, rf: float = 0.0
) -> float | pl.DataFrame:
    """Calculate absolute total excess return divided by absolute maximum drawdown."""
    returns_ldf = to_lazy(returns)
    expr = (RETURNS_COLUMNS_SELECTOR.sum() - rf).abs() / _drawdowns(
        RETURNS_COLUMNS_SELECTOR
    ).min().abs()
    return _metric_result(returns, returns_ldf, expr)


@overload
def ulcer_index(returns: pl.Series) -> float: ...


@overload
def ulcer_index(returns: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame: ...


def ulcer_index(returns: pl.Series | pl.DataFrame | pl.LazyFrame) -> float | pl.DataFrame:
    """Calculate root mean square drawdown."""
    returns_ldf = to_lazy(returns)
    dd = _drawdowns(RETURNS_COLUMNS_SELECTOR)
    expr = (dd.pow(2).sum() / (RETURNS_COLUMNS_SELECTOR.count() - 1)).sqrt()
    return _metric_result(returns, returns_ldf, expr)


@overload
def serenity_index(returns: pl.Series, rf: float = 0.0) -> float: ...


@overload
def serenity_index(returns: pl.DataFrame | pl.LazyFrame, rf: float = 0.0) -> pl.DataFrame: ...


def serenity_index(
    returns: pl.Series | pl.DataFrame | pl.LazyFrame, rf: float = 0.0
) -> float | pl.DataFrame:
    """Calculate Serenity Index using ulcer index and drawdown CVaR."""
    returns_ldf = to_lazy(returns)
    exprs: list[pl.Expr] = []
    z_score = _NORMAL.inv_cdf(0.05)
    for col_name in _numeric_column_names(returns_ldf):
        r = pl.col(col_name)
        dd = _drawdowns(r)
        dd_var = dd.mean() + z_score * dd.std(ddof=1)
        dd_cvar = dd.filter(dd < dd_var).mean().fill_null(dd_var)
        pitfall = -dd_cvar / r.std(ddof=1)
        ulcer = (dd.pow(2).sum() / (r.count() - 1)).sqrt()
        exprs.append(((r.sum() - rf) / (ulcer * pitfall)).alias(col_name))
    res = returns_ldf.select(exprs).collect()
    if isinstance(returns, pl.Series):
        return res.item()
    return res


def _drawdown_period_lengths(
    returns: pl.Series | pl.DataFrame | pl.LazyFrame,
) -> dict[str, list[int]]:
    returns_ldf = to_lazy(returns)
    temporal_names = cs.expand_selector(returns_ldf, cs.temporal())
    temporal_name = temporal_names[0] if len(temporal_names) == 1 else None
    df = returns_ldf.with_columns(_drawdowns(RETURNS_COLUMNS_SELECTOR)).collect()
    result: dict[str, list[int]] = {}
    date_values = df[temporal_name].to_list() if temporal_name else list(range(df.height))

    for col_name in _numeric_column_names(returns_ldf):
        lengths: list[int] = []
        start_idx: int | None = None
        for idx, value in enumerate(df[col_name].to_list()):
            in_drawdown = value is not None and value < 0
            if in_drawdown and start_idx is None:
                start_idx = idx
            if (not in_drawdown or idx == df.height - 1) and start_idx is not None:
                end_idx = idx if in_drawdown and idx == df.height - 1 else idx - 1
                start_value = date_values[start_idx]
                end_value = date_values[end_idx]
                if hasattr(end_value, "__sub__") and not isinstance(end_value, int):
                    lengths.append((end_value - start_value).days + 1)
                else:
                    lengths.append(end_idx - start_idx + 1)
                start_idx = None
        result[col_name] = lengths
    return result


def longest_drawdown_days(
    returns: pl.Series | pl.DataFrame | pl.LazyFrame,
) -> int | pl.DataFrame:
    """Calculate longest drawdown duration in days, or rows when no dates exist."""
    lengths = _drawdown_period_lengths(returns)
    values = {col: [max(col_lengths, default=0)] for col, col_lengths in lengths.items()}
    if isinstance(returns, pl.Series):
        return next(iter(values.values()))[0]
    return pl.DataFrame(values)


def avg_drawdown_days(returns: pl.Series | pl.DataFrame | pl.LazyFrame) -> float | pl.DataFrame:
    """Calculate average drawdown duration."""
    lengths = _drawdown_period_lengths(returns)
    values = {
        col: [sum(col_lengths) / len(col_lengths) if col_lengths else 0.0]
        for col, col_lengths in lengths.items()
    }
    if isinstance(returns, pl.Series):
        return next(iter(values.values()))[0]
    return pl.DataFrame(values)


def avg_drawdown(returns: pl.Series | pl.DataFrame | pl.LazyFrame) -> float | pl.DataFrame:
    """Calculate average maximum drawdown across drawdown periods."""
    returns_ldf = to_lazy(returns)
    df = returns_ldf.with_columns(_drawdowns(RETURNS_COLUMNS_SELECTOR)).collect()
    values: dict[str, list[float]] = {}
    for col_name in _numeric_column_names(returns_ldf):
        drawdowns: list[float] = []
        current: list[float] = []
        for value in df[col_name].to_list():
            if value is not None and value < 0:
                current.append(float(value))
            elif current:
                drawdowns.append(min(current))
                current = []
        if current:
            drawdowns.append(min(current))
        values[col_name] = [sum(drawdowns) / len(drawdowns) if drawdowns else 0.0]
    if isinstance(returns, pl.Series):
        return next(iter(values.values()))[0]
    return pl.DataFrame(values)


def risk_free_rate(rf: float, periods: int = 252) -> float:
    """Convert an annual risk-free rate to a per-period rate."""
    return (1 + rf) ** (1 / periods) - 1


def _date_filtered_returns(
    returns: pl.DataFrame | pl.LazyFrame, period: str, years: int | None = None
) -> pl.LazyFrame:
    returns_ldf = to_lazy(returns)
    temporal_name = _temporal_column_name(returns_ldf)
    max_date = returns_ldf.select(pl.col(temporal_name).max()).collect().item()
    temporal_col = pl.col(temporal_name)

    if period == "mtd":
        return returns_ldf.filter(
            (temporal_col.dt.year() == max_date.year) & (temporal_col.dt.month() == max_date.month)
        )
    if period == "ytd":
        return returns_ldf.filter(temporal_col.dt.year() == max_date.year)
    if period == "months":
        if years is None:
            raise ValueError("months period requires a month count")
        return returns_ldf.filter(temporal_col >= temporal_col.max().dt.offset_by(f"-{years}mo"))
    if period == "years":
        if years is None:
            raise ValueError("years period requires a year count")
        return returns_ldf.filter(temporal_col >= temporal_col.max().dt.offset_by(f"-{years}y"))
    return returns_ldf


def _period_comp(
    returns: pl.DataFrame | pl.LazyFrame, period: str, span: int | None = None
) -> pl.DataFrame:
    return (
        _date_filtered_returns(returns, period, span)
        .select(_comp(RETURNS_COLUMNS_SELECTOR))
        .collect()
    )


def _period_cagr(
    returns: pl.DataFrame | pl.LazyFrame, period: str, span: int | None = None, periods: int = 252
) -> pl.DataFrame:
    filtered = _date_filtered_returns(returns, period, span)
    temporal_col = get_temporal_column(filtered)
    if temporal_col is None:
        raise NoTemporalColumnError
    n_years = RETURNS_COLUMNS_SELECTOR.count() / periods
    return filtered.select(_comp(RETURNS_COLUMNS_SELECTOR).add(1).pow(1 / n_years).sub(1)).collect()


def mtd(returns: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    """Calculate month-to-date compounded return."""
    return _period_comp(returns, "mtd")


def three_month(returns: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    """Calculate trailing 3-month compounded return."""
    return _period_comp(returns, "months", 3)


def six_month(returns: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    """Calculate trailing 6-month compounded return."""
    return _period_comp(returns, "months", 6)


def ytd(returns: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    """Calculate year-to-date compounded return."""
    return _period_comp(returns, "ytd")


def one_year(returns: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    """Calculate trailing 1-year compounded return."""
    return _period_comp(returns, "years", 1)


def three_year(returns: pl.DataFrame | pl.LazyFrame, periods: int = 252) -> pl.DataFrame:
    """Calculate trailing 3-year annualized return."""
    return _period_cagr(returns, "years", 3, periods)


def five_year(returns: pl.DataFrame | pl.LazyFrame, periods: int = 252) -> pl.DataFrame:
    """Calculate trailing 5-year annualized return."""
    return _period_cagr(returns, "years", 5, periods)


def ten_year(returns: pl.DataFrame | pl.LazyFrame, periods: int = 252) -> pl.DataFrame:
    """Calculate trailing 10-year annualized return."""
    return _period_cagr(returns, "years", 10, periods)


def all_time(returns: pl.DataFrame | pl.LazyFrame, periods: int = 252) -> pl.DataFrame:
    """Calculate all-time annualized return."""
    return _period_cagr(returns, "all", periods=periods)


def expected_daily(returns: pl.Series | pl.DataFrame | pl.LazyFrame) -> float | pl.DataFrame:
    """Calculate expected daily return."""
    return expected_return(returns, aggregate="day")


def expected_monthly(
    returns: pl.Series | pl.DataFrame | pl.LazyFrame,
) -> float | pl.DataFrame:
    """Calculate expected monthly return."""
    return expected_return(returns, aggregate="month")


def expected_yearly(
    returns: pl.Series | pl.DataFrame | pl.LazyFrame,
) -> float | pl.DataFrame:
    """Calculate expected yearly return."""
    return expected_return(returns, aggregate="year")


def best_day(returns: pl.Series | pl.DataFrame | pl.LazyFrame) -> float | pl.DataFrame:
    """Return best daily return."""
    return best(returns, aggregate="day")


def worst_day(returns: pl.Series | pl.DataFrame | pl.LazyFrame) -> float | pl.DataFrame:
    """Return worst daily return."""
    return worst(returns, aggregate="day")


def best_month(returns: pl.Series | pl.DataFrame | pl.LazyFrame) -> float | pl.DataFrame:
    """Return best monthly return."""
    return best(returns, aggregate="month")


def worst_month(returns: pl.Series | pl.DataFrame | pl.LazyFrame) -> float | pl.DataFrame:
    """Return worst monthly return."""
    return worst(returns, aggregate="month")


def best_year(returns: pl.Series | pl.DataFrame | pl.LazyFrame) -> float | pl.DataFrame:
    """Return best yearly return."""
    return best(returns, aggregate="year")


def worst_year(returns: pl.Series | pl.DataFrame | pl.LazyFrame) -> float | pl.DataFrame:
    """Return worst yearly return."""
    return worst(returns, aggregate="year")


def avg_up_month(returns: pl.Series | pl.DataFrame | pl.LazyFrame) -> float | pl.DataFrame:
    """Calculate average positive monthly return."""
    return avg_win(returns, aggregate="month")


def avg_down_month(returns: pl.Series | pl.DataFrame | pl.LazyFrame) -> float | pl.DataFrame:
    """Calculate average negative monthly return."""
    return avg_loss(returns, aggregate="month")


def win_days(returns: pl.Series | pl.DataFrame | pl.LazyFrame) -> float | pl.DataFrame:
    """Calculate daily win rate."""
    return win_rate(returns, aggregate="day")


def win_month(returns: pl.Series | pl.DataFrame | pl.LazyFrame) -> float | pl.DataFrame:
    """Calculate monthly win rate."""
    return win_rate(returns, aggregate="month")


def win_quarter(returns: pl.Series | pl.DataFrame | pl.LazyFrame) -> float | pl.DataFrame:
    """Calculate quarterly win rate."""
    return win_rate(returns, aggregate="quarter")


def win_year(returns: pl.Series | pl.DataFrame | pl.LazyFrame) -> float | pl.DataFrame:
    """Calculate yearly win rate."""
    return win_rate(returns, aggregate="year")
