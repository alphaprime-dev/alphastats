import polars as pl
import polars.selectors as cs

RETURNS_COLUMNS_SELECTOR = cs.numeric()
DT_COLUMNS_SELECTOR = cs.temporal()


def get_dt_column(returns: pl.LazyFrame) -> pl.Expr:
    column_names = cs.expand_selector(returns, DT_COLUMNS_SELECTOR)

    if len(column_names) != 1:
        raise ValueError(f"Must have exactly one temporal column. Found {column_names}")

    return pl.col(column_names[0])


def to_lazy(returns: pl.Series | pl.DataFrame | pl.LazyFrame) -> pl.LazyFrame:
    match returns:
        case pl.Series():
            ldf = pl.LazyFrame(returns)
        case pl.DataFrame():
            ldf = returns.lazy()
        case pl.LazyFrame():
            ldf = returns

    return ldf.fill_nan(None)
