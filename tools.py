"""
tools.py - the analysis tools our agent can use.

Every tool is a plain Python function that works on ANY table (CSV).
Tools never mention a specific column name; the agent passes column names in
as arguments. That is what keeps the project from being "hard-coded".

The LLM never does maths itself. It only decides WHICH tool to call and with
WHAT arguments. pandas does the real calculation.
"""

import functools
import os
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # draw charts to files, no pop-up window needed
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Shared state: the dataset currently loaded
# ---------------------------------------------------------------------------
class _State:
    df: pd.DataFrame | None = None
    name: str = ""


STATE = _State()
CHART_DIR = "charts"


def load_dataset(path: str) -> str:
    """Load a CSV or Excel file so the tools can use it. Not an agent tool;
    our own code calls this when the user picks a file."""
    if path.lower().endswith((".xlsx", ".xls")):
        STATE.df = pd.read_excel(path)
    else:
        STATE.df = pd.read_csv(path)
    STATE.name = os.path.basename(path)
    return f"Loaded '{STATE.name}' with {len(STATE.df)} rows and {len(STATE.df.columns)} columns."


def _need_data():
    if STATE.df is None:
        raise ValueError("No dataset loaded yet.")
    return STATE.df


def _check_columns(*cols):
    df = _need_data()
    for c in cols:
        if c not in df.columns:
            raise ValueError(
                f"Column '{c}' does not exist. Available columns: {list(df.columns)}"
            )


def _safe(func):
    """Wrap a tool so errors come back as text the agent can read and fix,
    instead of crashing the program. functools.wraps keeps the original
    function's name, docstring and arguments, which the LLM SDK needs to
    describe the tool to the model."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            return {"error": str(e)}

    return wrapper


# ---------------------------------------------------------------------------
# TOOL 1: describe_dataset  (structure and data quality)
# ---------------------------------------------------------------------------
@_safe
def describe_dataset() -> dict:
    """Get an overview of the loaded dataset: number of rows, each column's
    name and type, missing values per column, duplicate rows, and 3 sample rows.
    Call this FIRST whenever you are unsure which columns exist."""
    df = _need_data()
    return {
        "rows": len(df),
        "columns": {c: str(t) for c, t in df.dtypes.items()},
        "missing_values": {c: int(n) for c, n in df.isna().sum().items() if n > 0},
        "duplicate_rows": int(df.duplicated().sum()),
        "sample_rows": df.head(3).to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# TOOL 2: summary_stats  (numeric distribution)
# ---------------------------------------------------------------------------
@_safe
def summary_stats(column: str) -> dict:
    """Statistics for ONE numeric column: count, mean, median, standard
    deviation, min, max, and quartiles (25%, 75%)."""
    _check_columns(column)
    s = pd.to_numeric(_need_data()[column], errors="coerce").dropna()
    if s.empty:
        raise ValueError(f"Column '{column}' is not numeric.")
    return {
        "column": column,
        "count": int(s.count()),
        "mean": round(float(s.mean()), 4),
        "median": round(float(s.median()), 4),
        "std": round(float(s.std()), 4),
        "min": float(s.min()),
        "25%": float(s.quantile(0.25)),
        "75%": float(s.quantile(0.75)),
        "max": float(s.max()),
    }


# ---------------------------------------------------------------------------
# TOOL 3: filter_rows  (selecting a subset)
# ---------------------------------------------------------------------------
@_safe
def filter_rows(column: str, operator: str, value: str) -> dict:
    """Find rows matching a condition and report how many match.
    operator must be one of: '==', '!=', '>', '<', '>=', '<=', 'contains'.
    value is given as text; it is converted to a number automatically when the
    column is numeric. Returns the match count and up to 5 example rows."""
    _check_columns(column)
    df = _need_data()
    col = df[column]

    if pd.api.types.is_numeric_dtype(col) and operator != "contains":
        value = float(value)

    if operator == "==":
        mask = col == value
    elif operator == "!=":
        mask = col != value
    elif operator == ">":
        mask = col > value
    elif operator == "<":
        mask = col < value
    elif operator == ">=":
        mask = col >= value
    elif operator == "<=":
        mask = col <= value
    elif operator == "contains":
        mask = col.astype(str).str.contains(str(value), case=False, na=False)
    else:
        raise ValueError("operator must be one of ==, !=, >, <, >=, <=, contains")

    matched = df[mask]
    return {
        "condition": f"{column} {operator} {value}",
        "matching_rows": int(len(matched)),
        "percent_of_total": round(100 * len(matched) / len(df), 2),
        "example_rows": matched.head(5).to_dict(orient="records"),
    }


# ---------------------------------------------------------------------------
# TOOL 4: group_and_aggregate  (compare categories)
# ---------------------------------------------------------------------------
@_safe
def group_and_aggregate(group_by: str, target_column: str, aggregation: str) -> dict:
    """Group rows by a category column and aggregate a numeric column.
    aggregation must be one of: 'sum', 'mean', 'count', 'min', 'max', 'median'.
    Example: group_by='region', target_column='sales', aggregation='sum' gives
    total sales per region, sorted from highest to lowest (top 15 shown)."""
    _check_columns(group_by, target_column)
    if aggregation not in {"sum", "mean", "count", "min", "max", "median"}:
        raise ValueError("aggregation must be sum, mean, count, min, max or median")
    df = _need_data()
    result = (
        df.groupby(group_by)[target_column]
        .agg(aggregation)
        .sort_values(ascending=False)
        .round(4)
    )
    return {
        "group_by": group_by,
        "target": target_column,
        "aggregation": aggregation,
        "number_of_groups": int(len(result)),
        "results_top_15": {str(k): float(v) for k, v in result.head(15).items()},
    }


# ---------------------------------------------------------------------------
# TOOL 5: correlation  (relationship between two columns)
# ---------------------------------------------------------------------------
@_safe
def correlation(column_a: str, column_b: str) -> dict:
    """Pearson correlation between two numeric columns, from -1 to +1.
    Near +1: they rise together. Near -1: one rises as the other falls.
    Near 0: no linear relationship."""
    _check_columns(column_a, column_b)
    df = _need_data()
    pair = df[[column_a, column_b]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(pair) < 3:
        raise ValueError("Not enough numeric data to compute a correlation.")
    r = float(pair[column_a].corr(pair[column_b]))
    strength = "strong" if abs(r) >= 0.7 else "moderate" if abs(r) >= 0.4 else "weak"
    direction = "positive" if r > 0 else "negative"
    return {
        "column_a": column_a,
        "column_b": column_b,
        "correlation": round(r, 4),
        "interpretation": f"{strength} {direction} relationship",
        "rows_used": int(len(pair)),
    }


# ---------------------------------------------------------------------------
# TOOL 6: plot_chart  (visual output)
# ---------------------------------------------------------------------------
@_safe
def plot_chart(chart_type: str, x_column: str, y_column: str = "") -> dict:
    """Draw a chart and save it as a PNG image.
    chart_type: 'bar' (average of y per x category), 'line' (y against x),
    'histogram' (distribution of x; y_column not needed), or 'scatter' (y vs x).
    Returns the file path of the saved image."""
    _check_columns(x_column)
    df = _need_data()
    os.makedirs(CHART_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))

    if chart_type == "histogram":
        df[x_column].plot(kind="hist", bins=20, ax=ax)
        ax.set_title(f"Distribution of {x_column}")
        ax.set_xlabel(x_column)
    else:
        if not y_column:
            raise ValueError(f"y_column is required for a {chart_type} chart.")
        _check_columns(y_column)
        if chart_type == "bar":
            df.groupby(x_column)[y_column].mean().sort_values(ascending=False).head(
                15
            ).plot(kind="bar", ax=ax)
            ax.set_title(f"Average {y_column} by {x_column}")
        elif chart_type == "line":
            df.sort_values(x_column).plot(x=x_column, y=y_column, kind="line", ax=ax)
            ax.set_title(f"{y_column} over {x_column}")
        elif chart_type == "scatter":
            df.plot(x=x_column, y=y_column, kind="scatter", ax=ax)
            ax.set_title(f"{y_column} vs {x_column}")
        else:
            raise ValueError("chart_type must be bar, line, histogram or scatter")
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)

    fig.tight_layout()
    safe_name = f"{chart_type}_{x_column}_{y_column}".replace(" ", "_").strip("_")
    path = os.path.join(CHART_DIR, f"{safe_name}.png")
    fig.savefig(path)
    plt.close(fig)
    return {"saved_to": path}


# The list the agent receives.
ALL_TOOLS = [
    describe_dataset,
    summary_stats,
    filter_rows,
    group_and_aggregate,
    correlation,
    plot_chart,
]
