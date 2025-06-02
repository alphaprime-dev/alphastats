import math
from collections.abc import Generator
from datetime import date

import polars as pl
import pytest
from inline_snapshot import snapshot

from alphastats import stats


@pytest.fixture(autouse=True)
def set_float_precision() -> Generator[None, None, None]:
    with pl.Config(set_float_precision=6):
        yield


@pytest.fixture
def simple_returns_series() -> pl.Series:
    """Simple returns series for basic testing."""
    return pl.Series("returns", [0.01, -0.02, 0.03, -0.01, 0.02])


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
            {"asset_a": [-0.9999999996618938], "asset_b": [-1.0]}
        )

    def test_cagr_with_risk_free_rate(self, simple_returns_df: pl.DataFrame) -> None:
        """Test CAGR calculation with risk-free rate."""
        result = stats.cagr(simple_returns_df, rf=0.02, periods=252)
        assert result.to_dict(as_series=False) == snapshot(
            {"asset_a": [-0.999999999411115], "asset_b": [-1.0]}
        )

    def test_cagr_non_compound(self, simple_returns_df: pl.DataFrame) -> None:
        """Test CAGR calculation without compounding."""
        result = stats.cagr(simple_returns_df, compound=False, periods=252)
        assert result.to_dict(as_series=False) == snapshot(
            {"asset_a": [-2.782176533738531e32], "asset_b": [-2.43521261277413e20]}
        )

    def test_cagr_different_periods(self, simple_returns_df: pl.DataFrame) -> None:
        """Test CAGR calculation with different period frequencies."""
        result = stats.cagr(simple_returns_df, periods=12)  # Monthly
        assert result.to_dict(as_series=False) == snapshot(
            {"asset_a": [-0.6460005937649241], "asset_b": [-0.9557500742206155]}
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
        assert result.to_dict(as_series=False) == snapshot({"asset": [-2.261248737261588e34]})


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
        assert result == snapshot(-3.06)

    def test_max_drawdown_dataframe_calculation(self, simple_returns_df: pl.DataFrame) -> None:
        """Test maximum drawdown calculation for dataframe."""
        result = stats.max_drawdown(simple_returns_df)
        assert result.to_dict(as_series=False) == snapshot(
            {"asset_a": [-3.06], "asset_b": [-1.5201500000000001]}
        )

    def test_max_drawdown_extreme_values(self, extreme_returns: pl.Series) -> None:
        """Test maximum drawdown with extreme values."""
        result = stats.max_drawdown(extreme_returns)
        assert result == snapshot(-2.6000000000000005)

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
        assert result == snapshot(-10.164280273197951)

    def test_sharpe_dataframe_calculation(self, simple_returns_df: pl.DataFrame) -> None:
        """Test Sharpe ratio calculation for dataframe."""
        result = stats.sharpe(simple_returns_df)
        assert result.to_dict(as_series=False) == snapshot(
            {"asset_a": [-10.164280273197951], "asset_b": [-11.20600940533624]}
        )

    def test_sharpe_with_risk_free_rate(self, simple_returns_series: pl.Series) -> None:
        """Test Sharpe ratio with risk-free rate."""
        result = stats.sharpe(simple_returns_series, rf=0.02)
        assert result == snapshot(-10.402133945737683)

    def test_sharpe_non_annualized(self, simple_returns_series: pl.Series) -> None:
        """Test Sharpe ratio without annualization."""
        result = stats.sharpe(simple_returns_series, annualize=False)
        assert result == snapshot(-0.640289472829558)

    def test_sharpe_different_periods(self, simple_returns_series: pl.Series) -> None:
        """Test Sharpe ratio with different period frequencies."""
        result = stats.sharpe(simple_returns_series, periods=12)  # Monthly
        assert result == snapshot(-2.218027796984573)

    def test_sharpe_extreme_values(self, extreme_returns: pl.Series) -> None:
        """Test Sharpe ratio with extreme values."""
        result = stats.sharpe(extreme_returns)
        assert result == snapshot(-12.740483884391793)

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
        assert result == snapshot(21.189498342339302)

    def test_volatility_dataframe_calculation(self, simple_returns_df: pl.DataFrame) -> None:
        """Test volatility calculation for dataframe."""
        result = stats.volatility(simple_returns_df)
        assert result.to_dict(as_series=False) == snapshot(
            {"asset_a": [21.189498342339302], "asset_b": [13.972502996957992]}
        )

    def test_volatility_non_annualized(self, simple_returns_series: pl.Series) -> None:
        """Test volatility without annualization."""
        result = stats.volatility(simple_returns_series, annualize=False)
        assert result == snapshot(1.3348129290486948)

    def test_volatility_different_periods(self, simple_returns_series: pl.Series) -> None:
        """Test volatility with different period frequencies."""
        result = stats.volatility(simple_returns_series, periods=12)  # Monthly
        assert result == snapshot(4.62392762342434)

    def test_volatility_extreme_values(self, extreme_returns: pl.Series) -> None:
        """Test volatility with extreme values."""
        result = stats.volatility(extreme_returns)
        assert result == snapshot(23.93315691671285)

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
            [0.0, -3.0, -3.06, -0.313333333333333, -0.29959999999999964]
        )

    def test_to_drawdowns_dataframe_calculation(self, simple_returns_df: pl.DataFrame) -> None:
        """Test drawdowns calculation for dataframe."""
        result = stats.to_drawdowns(simple_returns_df)
        expected_dict = result.select(pl.exclude("date")).to_dict(as_series=False)
        assert expected_dict == snapshot(
            {
                "asset_a": [0.0, -3.0, -3.06, -0.313333333333333, -0.29959999999999964],
                "asset_b": [0.0, -1.5, -1.505, -1.5201500000000001, -0.6532333333333333],
            }
        )

    def test_to_drawdowns_extreme_values(self, extreme_returns: pl.Series) -> None:
        """Test drawdowns with extreme values."""
        result = stats.to_drawdowns(extreme_returns)
        assert result.to_list() == snapshot(
            [0.0, -2.6000000000000005, 0.0, -1.7500000000000002, -1.9750000000000003]
        )

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
