import polars as pl
import polars.selectors as cs

from alphastats.exceptions import (
    AmbiguousBenchmarkReturnsError,
    MultipleTemporalColumnsError,
    NoReturnColumnError,
)

RETURNS_COLUMNS_SELECTOR = cs.numeric()
TEMPORAL_COLUMNS_SELECTOR = cs.temporal()

BENCHMARK_RETURNS_COLNAME = "_benchmark_returns"


def get_temporal_column(returns: pl.LazyFrame) -> pl.Expr | None:
    column_names = cs.expand_selector(returns, TEMPORAL_COLUMNS_SELECTOR)

    if len(column_names) > 1:
        raise MultipleTemporalColumnsError(column_names)

    return pl.col(column_names[0]) if column_names else None


def to_lazy(returns: pl.Series | pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    match returns:
        case pl.Series():
            ldf = pl.LazyFrame(returns)
        case pl.DataFrame():
            ldf = returns.lazy()
        case pl.LazyFrame():
            ldf = returns

    return ldf.fill_nan(None)


def to_excess_returns(expr: pl.Expr, rf: float | pl.Series | None) -> pl.Expr:
    if not rf:
        return expr

    return expr.sub(rf)


def prepare_benchmark(benchmark: pl.LazyFrame) -> pl.LazyFrame:
    column_names = cs.expand_selector(benchmark, RETURNS_COLUMNS_SELECTOR)
    match len(column_names):
        case 0:
            raise NoReturnColumnError
        case 1:
            return_col = pl.col(column_names[0])
        case _:
            raise AmbiguousBenchmarkReturnsError(column_names)

    if (temporal_col := get_temporal_column(benchmark)) is not None:
        return benchmark.select(temporal_col, return_col.alias(BENCHMARK_RETURNS_COLNAME))
    else:
        return benchmark.select(return_col.alias(BENCHMARK_RETURNS_COLNAME))
