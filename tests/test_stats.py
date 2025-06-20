import datetime
import math
from datetime import date

import polars as pl
import pytest
from inline_snapshot import snapshot

from alphastats import stats
from alphastats.exceptions import (
    AmbiguousBenchmarkReturnsError,
    MultipleTemporalColumnsError,
    NoReturnColumnError,
)


@pytest.fixture
def simple_returns_series() -> pl.Series:
    """Simple returns series for basic testing."""
    return pl.Series("returns", [0.01, -0.02, 0.03, -0.01, 0.02])


@pytest.fixture
def simple_benchmark_series() -> pl.Series:
    """Simple benchmark returns series for testing."""
    return pl.Series("_benchmark_returns", [0.005, -0.01, 0.015, -0.005, 0.01])


@pytest.fixture
def simple_returns_df() -> pl.DataFrame:
    """Simple returns dataframe with multiple assets."""
    return pl.DataFrame(
        {
            "date": [date(2023, 1, i) for i in range(1, 6)],
            "asset_a": [0.01, -0.02, 0.03, -0.01, 0.02],
            "asset_b": [0.02, -0.01, 0.01, 0.03, -0.02],
        }
    )


@pytest.fixture
def simple_benchmark_df() -> pl.DataFrame:
    """Simple benchmark dataframe with temporal column."""
    return pl.DataFrame(
        {
            "date": [date(2023, 1, i) for i in range(1, 6)],
            "_benchmark_returns": [0.005, -0.01, 0.015, -0.005, 0.01],
        }
    )


@pytest.fixture
def benchmark_different_dates() -> pl.DataFrame:
    """Benchmark data with different dates for asof join testing."""
    return pl.DataFrame(
        {
            "date": [date(2023, 1, i) for i in range(1, 8)],  # More dates than returns
            "_benchmark_returns": [0.005, -0.01, 0.015, -0.005, 0.01, 0.008, -0.003],
        }
    )


@pytest.fixture
def returns_with_nulls() -> pl.Series:
    """Returns series with null values."""
    return pl.Series("returns", [0.01, None, 0.03, -0.01, None])


@pytest.fixture
def extreme_returns() -> pl.Series:
    """Returns series with extreme values for edge case testing."""
    return pl.Series("returns", [0.5, -0.8, 1.2, -0.9, 0.3])


@pytest.fixture
def empty_returns() -> pl.Series:
    """Returns series with empty values for edge case testing."""
    return pl.Series("returns", [])


class TestComp:
    """Test cases for the comp (compound returns) function."""

    @pytest.mark.parametrize(
        "returns_fixture,expected_type",
        [
            ("simple_returns_series", float),
            ("simple_returns_df", pl.DataFrame),
        ],
    )
    def test_comp_return_types(
        self, returns_fixture: str, expected_type: type, request: pytest.FixtureRequest
    ) -> None:
        """Test that comp returns correct types for different inputs."""
        returns_data = request.getfixturevalue(returns_fixture)
        if returns_fixture == "simple_returns_df":
            # Test both DataFrame and LazyFrame
            result_df = stats.comp(returns_data)
            assert isinstance(result_df, expected_type)

            result_lazy = stats.comp(returns_data.lazy())
            assert isinstance(result_lazy, expected_type)
        else:
            result = stats.comp(returns_data)
            assert isinstance(result, expected_type)

    def test_comp_series_calculation(self, simple_returns_series: pl.Series) -> None:
        """Test compound returns calculation for series."""
        result = stats.comp(simple_returns_series)
        assert result == snapshot(0.02948504120000006)

    def test_comp_dataframe_calculation(self, simple_returns_df: pl.DataFrame) -> None:
        """Test compound returns calculation for dataframe."""
        result = stats.comp(simple_returns_df)
        assert result.to_dict(as_series=False) == snapshot(
            {"asset_a": [0.02948504120000006], "asset_b": [0.02948504120000006]}
        )

    def test_comp_with_nulls(self, returns_with_nulls: pl.Series) -> None:
        """Test compound returns with null values."""
        result = stats.comp(returns_with_nulls)
        assert result == snapshot(0.029897000000000062)

    def test_comp_extreme_values(self, extreme_returns: pl.Series) -> None:
        """Test compound returns with extreme values."""
        result = stats.comp(extreme_returns)
        assert result == snapshot(-0.9142)

    def test_comp_empty_data(self, empty_returns: pl.Series) -> None:
        """Test comp with empty data."""
        with pytest.raises(
            ValueError, match=r"can only call `.item\(\)` if the dataframe is of shape \(1, 1\)"
        ):
            stats.comp(empty_returns)

    def test_comp_single_value(self) -> None:
        """Test comp with single value."""
        single_series = pl.Series("returns", [0.05])
        result = stats.comp(single_series)
        assert result == snapshot(0.050000000000000044)

    def test_comp_all_zeros(self) -> None:
        """Test comp with all zero returns."""
        zero_series = pl.Series("returns", [0.0, 0.0, 0.0, 0.0])
        result = stats.comp(zero_series)
        assert result == snapshot(0.0)


class TestCagr:
    """Test cases for the CAGR (Compound Annual Growth Rate) function."""

    def test_cagr_basic_calculation(self, simple_returns_df: pl.DataFrame) -> None:
        """Test basic CAGR calculation."""
        result = stats.cagr(simple_returns_df, periods=252)
        assert result.to_dict(as_series=False) == snapshot(
            {"asset_a": [5.238246821747209], "asset_b": [5.238246821747209]}
        )

    def test_cagr_with_risk_free_rate(self, simple_returns_df: pl.DataFrame) -> None:
        """Test CAGR calculation with risk-free rate."""
        result = stats.cagr(simple_returns_df, rf=0.002, periods=252)
        assert result.to_dict(as_series=False) == snapshot(
            {"asset_a": [2.3321544328343586], "asset_b": [2.3321544328344044]}
        )

    def test_cagr_non_compound(self, simple_returns_df: pl.DataFrame) -> None:
        """Test CAGR calculation without compounding."""
        result = stats.cagr(simple_returns_df, compound=False, periods=252)
        assert result.to_dict(as_series=False) == snapshot(
            {"asset_a": [5.437913785074596], "asset_b": [5.437913785074596]}
        )

    def test_cagr_different_periods(self, simple_returns_df: pl.DataFrame) -> None:
        """Test CAGR calculation with different period frequencies."""
        result = stats.cagr(simple_returns_df, periods=12)  # Monthly
        assert result.to_dict(as_series=False) == snapshot(
            {"asset_a": [0.09108885990481008], "asset_b": [0.09108885990481008]}
        )

    def test_cagr_extreme_values(self) -> None:
        """Test CAGR with extreme values."""
        extreme_df = pl.DataFrame(
            {
                "date": [date(2023, 1, i) for i in range(1, 6)],
                "asset": [0.5, -0.8, 1.2, -0.9, 0.3],
            }
        )
        result = stats.cagr(extreme_df, periods=252)
        assert result.to_dict(as_series=False) == snapshot({"asset": [-1.0]})


class TestMaxDrawdown:
    """Test cases for the max_drawdown function."""

    @pytest.mark.parametrize(
        "returns_fixture,expected_type",
        [
            ("simple_returns_series", float),
            ("simple_returns_df", pl.DataFrame),
        ],
    )
    def test_max_drawdown_return_types(
        self, returns_fixture: str, expected_type: type, request: pytest.FixtureRequest
    ) -> None:
        """Test that max_drawdown returns correct types for different inputs."""
        returns_data = request.getfixturevalue(returns_fixture)
        if returns_fixture == "simple_returns_df":
            # Test both DataFrame and LazyFrame
            result_df = stats.max_drawdown(returns_data)
            assert isinstance(result_df, expected_type)

            result_lazy = stats.max_drawdown(returns_data.lazy())
            assert isinstance(result_lazy, expected_type)
        else:
            result = stats.max_drawdown(returns_data)
            assert isinstance(result, expected_type)

    def test_max_drawdown_series_calculation(self, simple_returns_series: pl.Series) -> None:
        """Test maximum drawdown calculation for series."""
        result = stats.max_drawdown(simple_returns_series)
        assert result == snapshot(-0.020000000000000018)

    def test_max_drawdown_dataframe_calculation(self, simple_returns_df: pl.DataFrame) -> None:
        """Test maximum drawdown calculation for dataframe."""
        result = stats.max_drawdown(simple_returns_df)
        assert result.to_dict(as_series=False) == snapshot(
            {"asset_a": [-0.020000000000000018], "asset_b": [-0.020000000000000018]}
        )

    def test_max_drawdown_extreme_values(self, extreme_returns: pl.Series) -> None:
        """Test maximum drawdown with extreme values."""
        result = stats.max_drawdown(extreme_returns)
        assert result == snapshot(-0.956)

    def test_max_drawdown_single_value(self) -> None:
        """Test max_drawdown with single value."""
        single_series = pl.Series("returns", [0.05])
        result = stats.max_drawdown(single_series)
        assert result == snapshot(0.0)

    def test_max_drawdown_all_zeros(self) -> None:
        """Test max_drawdown with all zero returns."""
        zero_series = pl.Series("returns", [0.0, 0.0, 0.0, 0.0])
        result = stats.max_drawdown(zero_series)
        assert result == snapshot(0.0)


class TestSharpe:
    """Test cases for the Sharpe ratio function."""

    @pytest.mark.parametrize(
        "returns_fixture,expected_type",
        [
            ("simple_returns_series", float),
            ("simple_returns_df", pl.DataFrame),
        ],
    )
    def test_sharpe_return_types(
        self, returns_fixture: str, expected_type: type, request: pytest.FixtureRequest
    ) -> None:
        """Test that sharpe returns correct types for different inputs."""
        returns_data = request.getfixturevalue(returns_fixture)
        if returns_fixture == "simple_returns_df":
            # Test both DataFrame and LazyFrame
            result_df = stats.sharpe(returns_data)
            assert isinstance(result_df, expected_type)

            result_lazy = stats.sharpe(returns_data.lazy())
            assert isinstance(result_lazy, expected_type)
        else:
            result = stats.sharpe(returns_data)
            assert isinstance(result, expected_type)

    def test_sharpe_series_calculation(self, simple_returns_series: pl.Series) -> None:
        """Test Sharpe ratio calculation for series."""
        result = stats.sharpe(simple_returns_series)
        assert result == snapshot(4.593220484431882)

    def test_sharpe_dataframe_calculation(self, simple_returns_df: pl.DataFrame) -> None:
        """Test Sharpe ratio calculation for dataframe."""
        result = stats.sharpe(simple_returns_df)
        assert result.to_dict(as_series=False) == snapshot(
            {"asset_a": [4.593220484431882], "asset_b": [4.593220484431882]}
        )

    def test_sharpe_with_risk_free_rate(self, simple_returns_series: pl.Series) -> None:
        """Test Sharpe ratio with risk-free rate."""
        result = stats.sharpe(simple_returns_series, rf=0.002)
        assert result == snapshot(3.062146989621255)

    def test_sharpe_non_annualized(self, simple_returns_series: pl.Series) -> None:
        """Test Sharpe ratio without annualization."""
        result = stats.sharpe(simple_returns_series, annualize=False)
        assert result == snapshot(0.28934569330224724)

    def test_sharpe_different_periods(self, simple_returns_series: pl.Series) -> None:
        """Test Sharpe ratio with different period frequencies."""
        result = stats.sharpe(simple_returns_series, periods=12)  # Monthly
        assert result == snapshot(1.002322883501468)

    def test_sharpe_extreme_values(self, extreme_returns: pl.Series) -> None:
        """Test Sharpe ratio with extreme values."""
        result = stats.sharpe(extreme_returns)
        assert result == snapshot(1.0629032821934614)

    def test_sharpe_all_zeros(self) -> None:
        """Test Sharpe ratio with all zero returns."""
        zero_series = pl.Series("returns", [0.0, 0.0, 0.0, 0.0])
        result = stats.sharpe(zero_series)
        assert math.isnan(result)


class TestVolatility:
    """Test cases for the volatility function."""

    @pytest.mark.parametrize(
        "returns_fixture,expected_type",
        [
            ("simple_returns_series", float),
            ("simple_returns_df", pl.DataFrame),
        ],
    )
    def test_volatility_return_types(
        self, returns_fixture: str, expected_type: type, request: pytest.FixtureRequest
    ) -> None:
        """Test that volatility returns correct types for different inputs."""
        returns_data = request.getfixturevalue(returns_fixture)
        if returns_fixture == "simple_returns_df":
            # Test both DataFrame and LazyFrame
            result_df = stats.volatility(returns_data)
            assert isinstance(result_df, expected_type)

            result_lazy = stats.volatility(returns_data.lazy())
            assert isinstance(result_lazy, expected_type)
        else:
            result = stats.volatility(returns_data)
            assert isinstance(result, expected_type)

    def test_volatility_series_calculation(self, simple_returns_series: pl.Series) -> None:
        """Test volatility calculation for series."""
        result = stats.volatility(simple_returns_series)
        assert result == snapshot(0.3291808013842849)

    def test_volatility_dataframe_calculation(self, simple_returns_df: pl.DataFrame) -> None:
        """Test volatility calculation for dataframe."""
        result = stats.volatility(simple_returns_df)
        assert result.to_dict(as_series=False) == snapshot(
            {"asset_a": [0.3291808013842849], "asset_b": [0.3291808013842849]}
        )

    def test_volatility_non_annualized(self, simple_returns_series: pl.Series) -> None:
        """Test volatility without annualization."""
        result = stats.volatility(simple_returns_series, annualize=False)
        assert result == snapshot(0.020736441353327723)

    def test_volatility_different_periods(self, simple_returns_series: pl.Series) -> None:
        """Test volatility with different period frequencies."""
        result = stats.volatility(simple_returns_series, periods=12)  # Monthly
        assert result == snapshot(0.07183313998427189)

    def test_volatility_extreme_values(self, extreme_returns: pl.Series) -> None:
        """Test volatility with extreme values."""
        result = stats.volatility(extreme_returns)
        assert result == snapshot(14.225188926689164)

    def test_volatility_all_zeros(self) -> None:
        """Test volatility with all zero returns."""
        zero_series = pl.Series("returns", [0.0, 0.0, 0.0, 0.0])
        result = stats.volatility(zero_series)
        assert result == snapshot(0.0)


class TestToDrawdowns:
    """Test cases for the to_drawdowns function."""

    @pytest.mark.parametrize(
        "returns_fixture,expected_type",
        [
            ("simple_returns_series", pl.Series),
            ("simple_returns_df", pl.DataFrame),
        ],
    )
    def test_to_drawdowns_return_types(
        self, returns_fixture: str, expected_type: type, request: pytest.FixtureRequest
    ) -> None:
        """Test that to_drawdowns returns correct types for different inputs."""
        returns_data = request.getfixturevalue(returns_fixture)
        if returns_fixture == "simple_returns_df":
            # Test both DataFrame and LazyFrame
            result_df = stats.to_drawdowns(returns_data)
            assert isinstance(result_df, expected_type)

            result_lazy = stats.to_drawdowns(returns_data.lazy())
            assert isinstance(result_lazy, expected_type)
        else:
            result = stats.to_drawdowns(returns_data)
            assert isinstance(result, expected_type)

    def test_to_drawdowns_series_calculation(self, simple_returns_series: pl.Series) -> None:
        """Test drawdowns calculation for series."""
        result = stats.to_drawdowns(simple_returns_series)
        assert result.to_list() == snapshot(
            [0.0, -0.020000000000000018, 0.0, -0.01000000000000012, 0.0]
        )

    def test_to_drawdowns_dataframe_calculation(self, simple_returns_df: pl.DataFrame) -> None:
        """Test drawdowns calculation for dataframe."""
        result = stats.to_drawdowns(simple_returns_df)
        expected_dict = result.to_dict(as_series=False)
        assert expected_dict == snapshot(
            {
                "date": [
                    datetime.date(2023, 1, 1),
                    datetime.date(2023, 1, 2),
                    datetime.date(2023, 1, 3),
                    datetime.date(2023, 1, 4),
                    datetime.date(2023, 1, 5),
                ],
                "asset_a": [0.0, -0.020000000000000018, 0.0, -0.01000000000000012, 0.0],
                "asset_b": [
                    0.0,
                    -0.010000000000000009,
                    -0.00010000000000010001,
                    0.0,
                    -0.020000000000000018,
                ],
            }
        )

    def test_to_drawdowns_extreme_values(self, extreme_returns: pl.Series) -> None:
        """Test drawdowns with extreme values."""
        result = stats.to_drawdowns(extreme_returns)
        assert result.to_list() == snapshot([0.0, -0.8, -0.56, -0.956, -0.9428])

    def test_to_drawdowns_all_zeros(self) -> None:
        """Test drawdowns with all zero returns."""
        zero_series = pl.Series("returns", [0.0, 0.0, 0.0, 0.0])
        result = stats.to_drawdowns(zero_series)
        assert result.to_list() == snapshot([0.0, 0.0, 0.0, 0.0])

    def test_to_drawdowns_single_value(self) -> None:
        """Test drawdowns with single value."""
        single_series = pl.Series("returns", [0.05])
        result = stats.to_drawdowns(single_series)
        assert result.to_list() == snapshot([0.0])


class TestGreeks:
    """Test cases for the greeks (CAPM alpha & beta) function."""

    def test_greeks_return_type(
        self, simple_returns_df: pl.DataFrame, simple_benchmark_df: pl.DataFrame
    ) -> None:
        """Test that greeks always returns a DataFrame."""
        # Test with DataFrame
        result_df = stats.greeks(simple_returns_df, simple_benchmark_df)
        assert isinstance(result_df, pl.DataFrame)

        # Test with LazyFrame
        result_lazy = stats.greeks(simple_returns_df.lazy(), simple_benchmark_df.lazy())
        assert isinstance(result_lazy, pl.DataFrame)

        # Test with Series for returns
        result_series = stats.greeks(
            simple_returns_df.select("asset_a").to_series(), simple_benchmark_df
        )
        assert isinstance(result_series, pl.DataFrame)

    def test_greeks_basic_calculation_with_temporal(
        self, simple_returns_df: pl.DataFrame, simple_benchmark_df: pl.DataFrame
    ) -> None:
        """Test basic greeks calculation with temporal columns."""
        result = stats.greeks(simple_returns_df, simple_benchmark_df)

        # Check that we have the expected columns
        assert "asset_a" in result.columns
        assert "asset_b" in result.columns

        # Check that result is a single row
        assert result.height == 1

        # Check that each column contains a struct with alpha and beta
        asset_a_struct = result["asset_a"][0]
        assert "alpha" in asset_a_struct
        assert "beta" in asset_a_struct
        assert isinstance(asset_a_struct["alpha"], float)
        assert isinstance(asset_a_struct["beta"], float)

    def test_greeks_calculation_without_temporal(self) -> None:
        """Test greeks calculation without temporal columns (horizontal concat)."""
        returns_no_date = pl.DataFrame(
            {
                "asset_a": [0.01, -0.02, 0.03, -0.01, 0.02],
                "asset_b": [0.02, -0.01, 0.01, 0.03, -0.02],
            }
        )
        benchmark_no_date = pl.DataFrame(
            {"_benchmark_returns": [0.005, -0.01, 0.015, -0.005, 0.01]}
        )

        result = stats.greeks(returns_no_date, benchmark_no_date)

        # Check basic structure
        assert isinstance(result, pl.DataFrame)
        assert result.height == 1
        assert "asset_a" in result.columns
        assert "asset_b" in result.columns

        assert result.to_dict(as_series=False) == snapshot(
            {
                "asset_a": [{"alpha": 2.185751579730777e-16, "beta": 1.9999999999999998}],
                "asset_b": [{"alpha": 1.6702325581395348, "beta": -0.20930232558139525}],
            }
        )

    def test_greeks_with_series_input(
        self, simple_returns_series: pl.Series, simple_benchmark_series: pl.Series
    ) -> None:
        """Test greeks with series inputs."""
        benchmark = simple_benchmark_series
        returns = simple_returns_series

        result = stats.greeks(returns, benchmark)

        assert isinstance(result, pl.DataFrame)
        assert result.height == 1
        assert "returns" in result.columns

        assert result.to_dict(as_series=False) == snapshot(
            {"returns": [{"alpha": 2.185751579730777e-16, "beta": 1.9999999999999998}]}
        )

    def test_greeks_asof_join_different_dates(
        self, simple_returns_df: pl.DataFrame, benchmark_different_dates: pl.DataFrame
    ) -> None:
        """Test greeks with asof join when benchmark has different/more dates."""
        result = stats.greeks(simple_returns_df, benchmark_different_dates)

        # Should still work with asof join
        assert isinstance(result, pl.DataFrame)
        assert result.height == 1
        assert "asset_a" in result.columns
        assert "asset_b" in result.columns

        assert result.to_dict(as_series=False) == snapshot(
            {
                "asset_a": [{"alpha": 2.185751579730777e-16, "beta": 1.9999999999999998}],
                "asset_b": [{"alpha": 1.6702325581395348, "beta": -0.20930232558139525}],
            }
        )

    def test_greeks_extreme_values(self) -> None:
        """Test greeks calculation with extreme values."""
        extreme_returns = pl.DataFrame(
            {
                "date": [date(2023, 1, i) for i in range(1, 6)],
                "asset": [0.5, -0.8, 1.2, -0.9, 0.3],
            }
        )
        extreme_benchmark = pl.DataFrame(
            {
                "date": [date(2023, 1, i) for i in range(1, 6)],
                "_benchmark_returns": [0.1, -0.2, 0.3, -0.15, 0.05],
            }
        )

        result = stats.greeks(extreme_returns, extreme_benchmark)

        assert isinstance(result, pl.DataFrame)
        assert result.height == 1
        assert "asset" in result.columns

        assert result.to_dict(as_series=False) == snapshot(
            {"asset": [{"alpha": -6.957055214723923, "beta": 4.380368098159508}]}
        )

    def test_greeks_single_asset(self) -> None:
        """Test greeks with single asset."""
        single_asset_returns = pl.DataFrame(
            {
                "date": [date(2023, 1, i) for i in range(1, 6)],
                "single_asset": [0.01, -0.02, 0.03, -0.01, 0.02],
            }
        )
        benchmark = pl.DataFrame(
            {
                "date": [date(2023, 1, i) for i in range(1, 6)],
                "_benchmark_returns": [0.005, -0.01, 0.015, -0.005, 0.01],
            }
        )

        result = stats.greeks(single_asset_returns, benchmark)

        assert isinstance(result, pl.DataFrame)
        assert result.height == 1
        assert len(result.columns) == 1
        assert "single_asset" in result.columns

        assert result.to_dict(as_series=False) == snapshot(
            {"single_asset": [{"alpha": 2.185751579730777e-16, "beta": 1.9999999999999998}]}
        )

    def test_greeks_zero_variance_benchmark(self) -> None:
        """Test greeks with zero variance benchmark (should handle division by zero)."""
        returns = pl.DataFrame(
            {
                "date": [date(2023, 1, i) for i in range(1, 6)],
                "asset": [0.01, -0.02, 0.03, -0.01, 0.02],
            }
        )
        zero_var_benchmark = pl.DataFrame(
            {
                "date": [date(2023, 1, i) for i in range(1, 6)],
                "_benchmark_returns": [0.01, 0.01, 0.01, 0.01, 0.01],  # No variance
            }
        )

        result = stats.greeks(returns, zero_var_benchmark)

        assert isinstance(result, pl.DataFrame)
        assert result.height == 1

        # Beta should be infinity when benchmark variance is zero
        asset_struct = result.item()
        assert math.isinf(asset_struct["beta"]) or math.isnan(asset_struct["beta"])

    def test_greeks_perfect_correlation(self) -> None:
        """Test greeks when returns and benchmark are perfectly correlated."""
        returns = pl.DataFrame(
            {
                "date": [date(2023, 1, i) for i in range(1, 6)],
                "asset": [0.02, -0.04, 0.06, -0.02, 0.04],  # 2x the benchmark
            }
        )
        benchmark = pl.DataFrame(
            {
                "date": [date(2023, 1, i) for i in range(1, 6)],
                "_benchmark_returns": [0.01, -0.02, 0.03, -0.01, 0.02],
            }
        )

        result = stats.greeks(returns, benchmark)

        assert isinstance(result, pl.DataFrame)
        assert result.height == 1

        asset_struct = result.item()
        # Beta should be close to 2.0 for perfectly correlated 2x returns
        assert abs(asset_struct["beta"] - 2.0) < 0.1

    def test_greeks_benchmark_no_numeric_columns_error(self) -> None:
        """Test greeks raises NoReturnColumnError when benchmark has no numeric columns."""
        returns = pl.DataFrame(
            {
                "date": [date(2023, 1, i) for i in range(1, 6)],
                "asset": [0.01, -0.02, 0.03, -0.01, 0.02],
            }
        )
        # Benchmark with only non-numeric columns
        invalid_benchmark = pl.DataFrame(
            {
                "date": [date(2023, 1, i) for i in range(1, 6)],
                "name": ["A", "B", "C", "D", "E"],  # String column, not numeric
            }
        )

        with pytest.raises(NoReturnColumnError):
            stats.greeks(returns, invalid_benchmark)

    def test_greeks_benchmark_multiple_numeric_columns_error(self) -> None:
        """
        Test greeks raises AmbiguousBenchmarkReturnsError when benchmark has multiple numeric
        columns
        """
        returns = pl.DataFrame(
            {
                "date": [date(2023, 1, i) for i in range(1, 6)],
                "asset": [0.01, -0.02, 0.03, -0.01, 0.02],
            }
        )
        # Benchmark with multiple numeric columns
        ambiguous_benchmark = pl.DataFrame(
            {
                "date": [date(2023, 1, i) for i in range(1, 6)],
                "benchmark_1": [0.005, -0.01, 0.015, -0.005, 0.01],
                "benchmark_2": [0.008, -0.012, 0.018, -0.008, 0.012],
            }
        )

        with pytest.raises(AmbiguousBenchmarkReturnsError):
            stats.greeks(returns, ambiguous_benchmark)

    def test_greeks_multiple_temporal_columns_error_returns(self) -> None:
        """
        Test greeks raises MultipleTemporalColumnsError when returns has multiple temporal columns
        """
        # Returns with multiple temporal columns
        invalid_returns = pl.DataFrame(
            {
                "date": [date(2023, 1, i) for i in range(1, 6)],
                "datetime": [datetime.datetime(2023, 1, i) for i in range(1, 6)],
                "asset": [0.01, -0.02, 0.03, -0.01, 0.02],
            }
        )
        benchmark = pl.DataFrame(
            {
                "date": [date(2023, 1, i) for i in range(1, 6)],
                "_benchmark_returns": [0.005, -0.01, 0.015, -0.005, 0.01],
            }
        )

        with pytest.raises(MultipleTemporalColumnsError):
            stats.greeks(invalid_returns, benchmark)

    def test_greeks_multiple_temporal_columns_error_benchmark(self) -> None:
        """
        Test greeks raises MultipleTemporalColumnsError when benchmark has multiple temporal columns
        """
        returns = pl.DataFrame(
            {
                "date": [date(2023, 1, i) for i in range(1, 6)],
                "asset": [0.01, -0.02, 0.03, -0.01, 0.02],
            }
        )
        # Benchmark with multiple temporal columns
        invalid_benchmark = pl.DataFrame(
            {
                "date": [date(2023, 1, i) for i in range(1, 6)],
                "datetime": [datetime.datetime(2023, 1, i) for i in range(1, 6)],
                "_benchmark_returns": [0.005, -0.01, 0.015, -0.005, 0.01],
            }
        )

        with pytest.raises(MultipleTemporalColumnsError):
            stats.greeks(returns, invalid_benchmark)
