from typing import overload

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

    n_years = (temporal_col.last() - temporal_col.first()).dt.total_days() / periods

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
    returns: pl.Series, rf: float = 0.0, periods: int = 252, annualize: bool = True
) -> float: ...


@overload
def sharpe(
    returns: pl.DataFrame | pl.LazyFrame,
    rf: float = 0.0,
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
