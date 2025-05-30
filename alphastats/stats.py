import polars as pl
import polars.selectors as cs

RETURNS_COLUMNS_SELECTOR = cs.numeric()
DT_COLUMNS_SELECTOR = cs.temporal()


def comp(returns: pl.Series | pl.DataFrame | pl.LazyFrame) -> float | pl.DataFrame:
    """
    Compute total compounded returns.

    Args:
        returns(pl.Series | pl.DataFrame | pl.LazyFrame): Returns series or dataframe

    Returns:
        float | pl.DataFrame: Compounded returns
    """
    if isinstance(returns, pl.Series):
        returns = pl.LazyFrame(returns)

    if isinstance(returns, pl.DataFrame):
        returns = returns.lazy()

    res = returns.select(_comp(RETURNS_COLUMNS_SELECTOR)).collect()

    if _has_single_returns_column(returns):
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
) -> float | pl.DataFrame:
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
        float | pl.DataFrame: CAGR%
    """

    if isinstance(returns, pl.DataFrame):
        returns = returns.lazy()

    returns = _prepare_returns(returns, rf)
    dt_col = _get_dt_column(returns)

    n_years = (dt_col.last() - dt_col.first()).dt.total_days() / periods

    if compound:
        expr = _comp(RETURNS_COLUMNS_SELECTOR).add(1).pow(1 / n_years).sub(1)
    else:
        expr = (RETURNS_COLUMNS_SELECTOR).sum().add(1).pow(1 / n_years).sub(1)

    return returns.with_columns(expr).collect()


# def max_drawdown(): ...


# def sharpe(): ...


# def volatility(): ...


# def greeks(): ...


# def to_drawdown_series(): ...


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


def _has_single_returns_column(returns: pl.LazyFrame) -> bool:
    return len(cs.expand_selector(returns, cs.numeric())) == 1


def _get_dt_column(returns: pl.LazyFrame) -> pl.Expr:
    column_names = cs.expand_selector(returns, DT_COLUMNS_SELECTOR)

    if len(column_names) != 1:
        raise ValueError(f"Must have exactly one temporal column. Found {column_names}")

    return DT_COLUMNS_SELECTOR
