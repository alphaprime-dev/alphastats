from collections.abc import Sequence


class AlphaStatsError(Exception):
    """Base exception for AlphaStats."""


class MultipleTemporalColumnsError(AlphaStatsError):
    """Exception raised when multiple temporal columns are found."""

    def __init__(self, column_names: Sequence[str]) -> None:
        self.column_names = column_names
        super().__init__(f"Must have exactly one temporal column. Found {column_names}")


class AmbiguousBenchmarkReturnsError(AlphaStatsError):
    """Exception raised when benchmark returns columns are ambiguous."""

    def __init__(self, column_names: Sequence[str]) -> None:
        self.column_names = column_names
        super().__init__(
            f"Ambiguous benchmark returns columns({column_names}). Please provide a dataframe"
            "with a single benchmark returns column."
        )


class NoTemporalColumnError(AlphaStatsError):
    """Exception raised when no temporal column is found."""

    def __init__(self) -> None:
        super().__init__(
            "This function requires a temporal column. Please provide a dataframe with a"
            "temporal column."
        )


class NoReturnColumnError(AlphaStatsError):
    """Exception raised when no return column is found."""

    def __init__(self) -> None:
        super().__init__("No return column found. Please provide a dataframe with areturn column.")
