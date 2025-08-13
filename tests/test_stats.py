import datetime
import math
from datetime import date
from typing import cast

import polars as pl
import pytest
from inline_snapshot import snapshot

from alphastats import stats
from alphastats.exceptions import (
    AmbiguousBenchmarkReturnsError,
    MultipleTemporalColumnsError,
    NoReturnColumnError,
    NoTemporalColumnError,
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


class TestProbabilisticSharpeRatio:
    """Tests for the Probabilistic Sharpe Ratio (PSR)."""

    @pytest.mark.parametrize(
        "returns_fixture,expected_type",
        [
            ("simple_returns_series", float),
            ("simple_returns_df", pl.DataFrame),
        ],
    )
    def test_psr_return_types(
        self, returns_fixture: str, expected_type: type, request: pytest.FixtureRequest
    ) -> None:
        data = request.getfixturevalue(returns_fixture)
        if returns_fixture == "simple_returns_df":
            assert isinstance(stats.probabilistic_sharpe_ratio(data), expected_type)
            assert isinstance(stats.psr(data), expected_type)
            assert isinstance(stats.probabilistic_sharpe_ratio(data.lazy()), expected_type)
        else:
            assert isinstance(stats.probabilistic_sharpe_ratio(data), expected_type)
            assert isinstance(stats.psr(data), expected_type)

    def test_psr_basic_series(self, simple_returns_series: pl.Series) -> None:
        # Deterministic value based on fixed inputs
        val = stats.probabilistic_sharpe_ratio(simple_returns_series)
        assert val == snapshot(0.7132960099383969)

    def test_psr_with_benchmark(self, simple_returns_series: pl.Series) -> None:
        # With a higher benchmark SR, probability should decrease
        base = stats.psr(simple_returns_series, sr_benchmark=0.0)
        lower = stats.psr(simple_returns_series, sr_benchmark=1.0)
        assert lower <= base

    def test_psr_dataframe_values(self, simple_returns_df: pl.DataFrame) -> None:
        res = stats.psr(simple_returns_df)
        assert isinstance(res, pl.DataFrame)
        assert set(res.columns) == {"asset_a", "asset_b"}
        for col in res.columns:
            assert 0.0 <= res[col][0] <= 1.0


class TestSortino:
    """Test cases for the Sortino ratio function."""

    @pytest.mark.parametrize(
        "returns_fixture,expected_type",
        [
            ("simple_returns_series", float),
            ("simple_returns_df", pl.DataFrame),
        ],
    )
    def test_sortino_return_types(
        self, returns_fixture: str, expected_type: type, request: pytest.FixtureRequest
    ) -> None:
        """Test that sortino returns correct types for different inputs."""
        returns_data = request.getfixturevalue(returns_fixture)
        if returns_fixture == "simple_returns_df":
            # Test both DataFrame and LazyFrame
            result_df = stats.sortino(returns_data)
            assert isinstance(result_df, expected_type)

            result_lazy = stats.sortino(returns_data.lazy())
            assert isinstance(result_lazy, expected_type)
        else:
            result = stats.sortino(returns_data)
            assert isinstance(result, expected_type)

    def test_sortino_series_calculation(self, simple_returns_series: pl.Series) -> None:
        """Test Sortino ratio calculation for series (annualized)."""
        result = stats.sortino(simple_returns_series)
        assert result == snapshot(9.524704719832526)

    def test_sortino_dataframe_calculation(self, simple_returns_df: pl.DataFrame) -> None:
        """Test Sortino ratio calculation for dataframe (annualized)."""
        result = stats.sortino(simple_returns_df)
        assert result.to_dict(as_series=False) == snapshot(
            {"asset_a": [9.524704719832526], "asset_b": [9.524704719832526]}
        )

    def test_sortino_with_risk_free_rate_non_annualized(
        self, simple_returns_series: pl.Series
    ) -> None:
        """Test Sortino ratio with risk-free rate without annualization."""
        result = stats.sortino(simple_returns_series, rf=0.002, annualize=False)
        assert result == snapshot(0.35691530512412484)

    def test_sortino_non_annualized(self, simple_returns_series: pl.Series) -> None:
        """Test Sortino ratio without annualization."""
        result = stats.sortino(simple_returns_series, annualize=False)
        assert result == snapshot(0.6)

    def test_sortino_different_periods(self, simple_returns_series: pl.Series) -> None:
        """Test Sortino ratio with different period frequency (monthly)."""
        result = stats.sortino(simple_returns_series, periods=12)
        assert result == snapshot(2.0784609690826525)

    def test_sortino_extreme_values(self, extreme_returns: pl.Series) -> None:
        """Test Sortino ratio with extreme values (annualized)."""
        result = stats.sortino(extreme_returns)
        assert result == snapshot(1.7686932639858621)

    def test_sortino_all_zeros(self) -> None:
        """Test Sortino ratio with all zero returns (NaN due to zero downside risk)."""
        zero_series = pl.Series("returns", [0.0, 0.0, 0.0, 0.0])
        result = stats.sortino(zero_series)
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


class TestCalmar:
    """Tests for the Calmar ratio metric."""

    def test_calmar_basic(self, simple_returns_df: pl.DataFrame) -> None:
        # given
        cagr_df = stats.cagr(simple_returns_df, periods=252)
        mdd_df = stats.max_drawdown(simple_returns_df)
        expected = {col: [cagr_df[col][0] / abs(mdd_df[col][0])] for col in ["asset_a", "asset_b"]}

        # when
        result = stats.calmar(simple_returns_df, periods=252)

        # then
        assert result.to_dict(as_series=False) == expected

    def test_calmar_extreme(self) -> None:
        # given
        df = pl.DataFrame(
            {
                "date": [date(2023, 1, i) for i in range(1, 6)],
                "asset": [0.5, -0.8, 1.2, -0.9, 0.3],
            }
        )
        cagr_val = stats.cagr(df, periods=252)["asset"][0]
        mdd_val = abs(stats.max_drawdown(df)["asset"][0])
        expected_val = cagr_val / mdd_val

        # when
        result = stats.calmar(df, periods=252)

        # then
        res_dict = result.to_dict(as_series=False)
        assert pytest.approx(res_dict["asset"][0], rel=1e-3) == expected_val

    def test_calmar_requires_temporal_column(self) -> None:
        df = pl.DataFrame({"asset": [0.01, -0.02, 0.03, -0.01, 0.02]})
        with pytest.raises(MultipleTemporalColumnsError):
            # Ensure no confusion with other errors; add a second temporal column scenario
            invalid = pl.DataFrame(
                {
                    "date": [date(2023, 1, i) for i in range(1, 6)],
                    "datetime": [datetime.datetime(2023, 1, i) for i in range(1, 6)],
                    "asset": [0.01, -0.02, 0.03, -0.01, 0.02],
                }
            )
            stats.calmar(invalid)

        with pytest.raises(NoTemporalColumnError):
            # Without any temporal column, should raise NoTemporalColumnError
            stats.calmar(df)


class TestCPCIndex:
    """Tests for the CPC Index metric."""

    @pytest.mark.parametrize(
        "returns_fixture,expected_type",
        [
            ("simple_returns_series", float),
            ("simple_returns_df", pl.DataFrame),
        ],
    )
    def test_cpc_index_return_types(
        self, returns_fixture: str, expected_type: type, request: pytest.FixtureRequest
    ) -> None:
        data = request.getfixturevalue(returns_fixture)
        if returns_fixture == "simple_returns_df":
            assert isinstance(stats.cpc_index(data), expected_type)
            assert isinstance(stats.cpc_index(data.lazy()), expected_type)
        else:
            assert isinstance(stats.cpc_index(data), expected_type)

    def test_cpc_index_simple_series(self, simple_returns_series: pl.Series) -> None:
        # Manual computation on fixture [0.01, -0.02, 0.03, -0.01, 0.02]
        # gains_sum = 0.01 + 0.03 + 0.02 = 0.06
        # losses_sum_abs = | -0.02 + -0.01 | = 0.03
        # wins_count = 3, losses_count = 2, total = 5
        # avg_win = 0.06 / 3 = 0.02
        # avg_loss_abs = 0.03 / 2 = 0.015
        # profit_factor = 0.06 / 0.03 = 2.0
        # payoff_ratio = 0.02 / 0.015 = 1.3333333333333333
        # win_rate = 3 / 5 = 0.6
        # CPC = 2.0 * 1.3333333333333333 * 0.6 = 1.6
        result = stats.cpc_index(simple_returns_series)
        assert result == 1.6

    def test_cpc_index_dataframe_values(self, simple_returns_df: pl.DataFrame) -> None:
        # asset_b has same distribution as a; CPC should match
        res = stats.cpc_index(simple_returns_df)
        assert res.to_dict(as_series=False) == snapshot({"asset_a": [1.6], "asset_b": [1.6]})

    def test_cpc_index_with_nulls(self, returns_with_nulls: pl.Series) -> None:
        # returns_with_nulls: [0.01, None, 0.03, -0.01, None]
        # gains_sum = 0.04, losses_sum_abs = 0.01
        # wins = 2, losses = 1, total = 3
        # avg_win = 0.02, avg_loss_abs = 0.01
        # PF=4.0, PR=2.0, WR=2/3 -> CPC = 16/3 = 5.333333333333333
        val = stats.cpc_index(returns_with_nulls)
        assert val == 5.333333333333333

    def test_cpc_index_edge_no_losses(self) -> None:
        s = pl.Series("returns", [0.01, 0.02, 0.0])
        # losses_sum_abs = 0 → PF and PR divisions yield inf/NaN; product should be NaN
        res = stats.cpc_index(s)
        assert math.isnan(res) or math.isinf(res)

    def test_cpc_index_edge_no_wins(self) -> None:
        s = pl.Series("returns", [-0.01, -0.02, 0.0])
        res = stats.cpc_index(s)
        assert math.isnan(res) or math.isinf(res)


class TestExposure:
    """Tests for Time in Market (exposure)."""

    @pytest.mark.parametrize(
        "returns_fixture,expected_type",
        [
            ("simple_returns_series", float),
            ("simple_returns_df", pl.DataFrame),
        ],
    )
    def test_exposure_return_types(
        self, returns_fixture: str, expected_type: type, request: pytest.FixtureRequest
    ) -> None:
        data = request.getfixturevalue(returns_fixture)
        if returns_fixture == "simple_returns_df":
            assert isinstance(stats.exposure(data), expected_type)
            assert isinstance(stats.exposure(data.lazy()), expected_type)
        else:
            assert isinstance(stats.exposure(data), expected_type)

    def test_exposure_simple_series(self, simple_returns_series: pl.Series) -> None:
        # [0.01, -0.02, 0.03, -0.01, 0.02] all non-zero -> 5/5 = 1.0
        assert stats.exposure(simple_returns_series) == 1.0

    def test_exposure_dataframe_values(self, simple_returns_df: pl.DataFrame) -> None:
        # both columns are fully non-zero
        res = stats.exposure(simple_returns_df)
        assert res.to_dict(as_series=False) == {"asset_a": [1.0], "asset_b": [1.0]}

    def test_exposure_with_nulls(self, returns_with_nulls: pl.Series) -> None:
        # [0.01, None, 0.03, -0.01, None] -> non-zero count=3, total non-null=3 => 1.0
        assert stats.exposure(returns_with_nulls) == 1.0

    def test_exposure_with_zeros(self) -> None:
        s = pl.Series("returns", [0.0, 0.01, 0.0, -0.02, 0.0, 0.03])
        # non-zero = 3, total = 6 => 0.5
        assert stats.exposure(s) == 0.5


class TestOmega:
    """Tests for Omega ratio."""

    @pytest.mark.parametrize(
        "returns_fixture,expected_type",
        [
            ("simple_returns_series", float),
            ("simple_returns_df", pl.DataFrame),
        ],
    )
    def test_omega_return_types(
        self, returns_fixture: str, expected_type: type, request: pytest.FixtureRequest
    ) -> None:
        data = request.getfixturevalue(returns_fixture)
        if returns_fixture == "simple_returns_df":
            assert isinstance(stats.omega(data), expected_type)
            assert isinstance(stats.omega(data.lazy()), expected_type)
        else:
            assert isinstance(stats.omega(data), expected_type)

    def test_omega_simple_series_default_threshold(self, simple_returns_series: pl.Series) -> None:
        # r = [0.01, -0.02, 0.03, -0.01, 0.02]
        # θ=0.0: gains=sum([0.01,0.03,0.02])=0.06, losses=sum([0.02,0.01])=0.03 → 2.0
        assert stats.omega(simple_returns_series) == 2.0

    def test_omega_series_custom_threshold(self) -> None:
        # θ=0.01: (r-θ)+ sums: [0,0,0.02,0,0.01] => 0.03
        # (θ-r)+ sums: [0,0.03,0,0.02,0] => 0.05 → 0.6
        s = pl.Series("returns", [0.01, -0.02, 0.03, -0.01, 0.02])
        assert stats.omega(s, threshold=0.01) == 0.6

    def test_omega_dataframe_values(self, simple_returns_df: pl.DataFrame) -> None:
        res = stats.omega(simple_returns_df)
        assert res.to_dict(as_series=False) == {"asset_a": [2.0], "asset_b": [2.0]}

    def test_omega_with_nulls(self, returns_with_nulls: pl.Series) -> None:
        # [0.01, None, 0.03, -0.01, None]
        # gains=0.04, losses=0.01 → 4.0
        assert stats.omega(returns_with_nulls) == 4.0

    def test_omega_edge_no_losses(self) -> None:
        s = pl.Series("returns", [0.02, 0.03, 0.01])
        val = stats.omega(s)
        assert math.isinf(val) or math.isnan(val)


class TestInformationRatio:
    """Tests for Information Ratio."""

    @pytest.mark.parametrize(
        "returns_fixture,expected_type",
        [
            ("simple_returns_series", float),
            ("simple_returns_df", pl.DataFrame),
        ],
    )
    def test_ir_return_types(
        self,
        returns_fixture: str,
        expected_type: type,
        request: pytest.FixtureRequest,
        simple_benchmark_df: pl.DataFrame,
        simple_benchmark_series: pl.Series,
    ) -> None:
        data = request.getfixturevalue(returns_fixture)
        if returns_fixture == "simple_returns_df":
            assert isinstance(stats.information_ratio(data, simple_benchmark_df), expected_type)
            assert isinstance(
                stats.information_ratio(data.lazy(), simple_benchmark_df.lazy()), expected_type
            )
        else:
            assert isinstance(stats.information_ratio(data, simple_benchmark_series), expected_type)

    def test_ir_simple_series(
        self, simple_returns_series: pl.Series, simple_benchmark_series: pl.Series
    ) -> None:
        # given
        # active = [0.005, -0.01, 0.015, -0.005, 0.01]
        # mean=0.003, std(sample)=? compute explicitly
        active = simple_returns_series - simple_benchmark_series
        expected = cast(float, active.mean()) / cast(float, active.std(ddof=1)) * (252**0.5)

        # when
        val = stats.information_ratio(simple_returns_series, simple_benchmark_series)

        # then
        assert pytest.approx(val, rel=1e-3) == expected

    def test_ir_dataframe_values(
        self, simple_returns_df: pl.DataFrame, simple_benchmark_df: pl.DataFrame
    ) -> None:
        # given
        # asset_a
        active_a = simple_returns_df["asset_a"] - simple_benchmark_df["_benchmark_returns"]
        expected_a = cast(float, active_a.mean()) / cast(float, active_a.std(ddof=1)) * (252**0.5)
        # asset_b
        active_b = simple_returns_df["asset_b"] - simple_benchmark_df["_benchmark_returns"]
        expected_b = cast(float, active_b.mean()) / cast(float, active_b.std(ddof=1)) * (252**0.5)

        # when
        res = stats.information_ratio(simple_returns_df, simple_benchmark_df)

        # then
        res_dict = res.to_dict(as_series=False)
        assert pytest.approx(res_dict["asset_a"][0], rel=1e-3) == expected_a
        assert pytest.approx(res_dict["asset_b"][0], rel=1e-3) == expected_b

    def test_ir_non_annualized(
        self, simple_returns_series: pl.Series, simple_benchmark_series: pl.Series
    ) -> None:
        # given
        active = simple_returns_series - simple_benchmark_series
        expected = cast(float, active.mean()) / cast(float, active.std(ddof=1))

        # when
        val = stats.information_ratio(
            simple_returns_series, simple_benchmark_series, annualize=False
        )

        # then
        assert pytest.approx(val, rel=1e-3) == expected
