from typing import overload

import polars as pl
import polars.selectors as cs

RETURNS_COLUMNS_SELECTOR = cs.numeric()
DT_COLUMNS_SELECTOR = cs.temporal()


@overload
def comp(returns: pl.Series) -> float: ...


@overload
def comp(returns: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame: ...


def comp(returns: pl.Series | pl.DataFrame | pl.LazyFrame) -> float | pl.DataFrame:
    """
    Compute total compounded returns.

    Args:
        returns(pl.Series | pl.DataFrame | pl.LazyFrame): Returns series or dataframe

    Returns:
        float | pl.DataFrame: Compounded returns
    """
    returns_ldf = _to_lazy(returns)

    res = returns_ldf.select(_comp(RETURNS_COLUMNS_SELECTOR)).collect()

    if isinstance(returns, pl.Series):
        return res.item()
    else:
        return res


def _comp(expr: pl.Expr) -> pl.Expr:
    return (expr + 1).product() - 1


def cagr(
    returns: pl.DataFrame | pl.LazyFrame,
    rf: float = 0.0,
    compound: bool = True,
    periods: int = 252,
) -> pl.DataFrame:
    """
    Calculates the communicative annualized growth return
    (CAGR%) of access returns

    If rf is non-zero, you must specify periods.
    In this case, rf is assumed to be expressed in yearly (annualized) terms

    Args:
        returns(pl.DataFrame | pl.LazyFrame): Returns dataframe or lazyframe
        rf(float): Risk-free rate
        compound(bool): Whether to compound the returns
        periods(int): Number of periods in a year

    Returns:
        pl.DataFrame: CAGR%
    """
    returns_ldf = _prepare_returns(_to_lazy(returns), rf)
    dt_col = _get_dt_column(returns_ldf)

    n_years = (dt_col.last() - dt_col.first()).dt.total_days() / periods

    if compound:
        expr = _comp(RETURNS_COLUMNS_SELECTOR).add(1).pow(1 / n_years).sub(1)
    else:
        expr = (RETURNS_COLUMNS_SELECTOR).sum().add(1).pow(1 / n_years).sub(1)

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
        returns(pl.Series | pl.DataFrame | pl.LazyFrame): Returns series or dataframe

    Returns:
        float | pl.DataFrame: Maximum drawdown value(s)
    """
    returns_ldf = _prepare_returns(_to_lazy(returns))

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
    Calculates the Sharpe ratio of access returns.

    The Sharpe ratio measures risk-adjusted return by dividing excess return by volatility.

    Args:
        returns(pl.Series | pl.DataFrame | pl.LazyFrame): Returns series or dataframe
        rf(float): Risk-free rate expressed as yearly (annualized) return
        periods(int): Frequency of returns (252 for daily, 12 for monthly)
        annualize(bool): Whether to annualize the Sharpe ratio

    Returns:
        float | pl.DataFrame: Sharpe ratio value(s)
    """
    returns_ldf = _prepare_returns(_to_lazy(returns), rf)

    sharpe = RETURNS_COLUMNS_SELECTOR.mean() / RETURNS_COLUMNS_SELECTOR.std(ddof=1)

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
        returns(pl.Series | pl.DataFrame | pl.LazyFrame): Returns series or dataframe
        periods(int): Frequency of returns (252 for daily, 12 for monthly)
        annualize(bool): Whether to annualize the volatility

    Returns:
        float | pl.DataFrame: Volatility value(s)
    """
    returns_ldf = _prepare_returns(_to_lazy(returns))

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
        returns(pl.Series | pl.DataFrame | pl.LazyFrame): Returns series or dataframe

    Returns:
        pl.Series | pl.DataFrame: Drawdowns with same structure as input
    """
    returns_ldf = _prepare_returns(_to_lazy(returns))

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


def _prepare_returns(returns: pl.LazyFrame, rf: float | pl.Series | None = None) -> pl.LazyFrame:
    expr = (
        pl.when((RETURNS_COLUMNS_SELECTOR).drop_nulls().is_between(0, 1))
        .then(RETURNS_COLUMNS_SELECTOR)
        .otherwise((RETURNS_COLUMNS_SELECTOR).pct_change())
        .fill_null(0)
    )

    if rf:
        return returns.with_columns(expr.sub(rf))

    return returns.with_columns(expr)


def _get_dt_column(returns: pl.LazyFrame) -> pl.Expr:
    column_names = cs.expand_selector(returns, DT_COLUMNS_SELECTOR)

    if len(column_names) != 1:
        raise ValueError(f"Must have exactly one temporal column. Found {column_names}")

    return pl.col(column_names[0])


def _to_lazy(returns: pl.Series | pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    match returns:
        case pl.Series():
            ldf = pl.LazyFrame(returns)
        case pl.DataFrame():
            ldf = returns.lazy()
        case pl.LazyFrame():
            ldf = returns

    return ldf.fill_nan(None)
