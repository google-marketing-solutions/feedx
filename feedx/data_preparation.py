# Copyright 2023 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data preparation for FeedX.

This module contains functions for loading and preparing the data required for
FeedX.
"""

from collections.abc import Collection
import datetime as dt
from typing import Callable
import numpy as np
import numpy.typing as npt
import pandas as pd

GOOGLE_ADS_PERFORMANCE_CSV_COLUMNS = {
    "date_column": "Week",
    "item_id_column": "Item ID",
    "impressions_column": "Impr.",
    "clicks_column": "Clicks",
}
GOOGLE_ADS_PERFORMANCE_READ_CSV_ARGS = {"header": 2, "thousands": ","}


def _sample_from_lognormal(
    rng: np.random.Generator,
    mean: npt.ArrayLike,
    standard_deviation: npt.ArrayLike,
    size: int | tuple[int, int],
) -> np.ndarray:
  """Samples from a lognormal distribution."""
  valid_mean = mean > 0
  valid_standard_deviation = standard_deviation > 0
  if not np.all(valid_mean):
    raise ValueError("mean must be greater than 0")
  if not np.all(valid_standard_deviation):
    raise ValueError("standard_deviation must be greater than 0")

  lognormal_mean = np.log(mean) - 0.5 * np.log(
      standard_deviation**2 / mean**2 + 1
  )
  lognormal_sigma = np.sqrt(np.log(standard_deviation**2 / mean**2 + 1))
  return rng.lognormal(mean=lognormal_mean, sigma=lognormal_sigma, size=size)


def _sample_from_beta(
    rng: np.random.Generator,
    mean: npt.ArrayLike,
    standard_deviation: npt.ArrayLike,
    size: int | tuple[int, int],
) -> np.ndarray:
  """Samples from a beta distribution."""
  valid_mean = 0 < mean < 1
  valid_standard_deviation = 0 < standard_deviation < 1
  if not np.all(valid_mean):
    raise ValueError("mean must be between 0 and 1")
  if not np.all(valid_standard_deviation):
    raise ValueError("standard_deviation must be between 0 and 1")

  a_plus_b = np.sqrt(mean * (1.0 - mean)) / standard_deviation
  return rng.beta(
      a=mean * a_plus_b,
      b=(1.0 - mean) * a_plus_b,
      size=size,
  )


def generate_synthetic_data(
    rng: np.random.Generator,
    n_items: int = 7000,
    impressions_average: float = 100.0,
    impressions_standard_deviation: float = 100.0,
    ctr_average: float = 0.02,
    ctr_standard_deviation: float = 0.015,
    cpc_average: float = 0.5,
    cpc_item_level_standard_deviation: float = 0.5,
    cpc_daily_level_standard_deviation: float = 0.5,
    conversion_rate_average: float = 0.1,
    conversion_rate_standard_deviation: float = 0.075,
    conversion_value_average: float = 100.0,
    conversion_value_item_level_standard_deviation: float = 80.0,
    conversion_value_daily_level_standard_deviation: float = 10.0,
    weeks_before_start_date: int = 0,
    weeks_after_start_date: int = 13,
    start_date: str = "2023-01-01",
) -> pd.DataFrame:
  """Generates synthetic data for demonstrating the FeedX notebooks.

  This generates daily clicks, impressions, cost, conversions and conversion
  value data which can be used to demonstrate how FeedX designs and analyzes
  experiments. There is no "effect" in the data, all items are drawn from the
  same distribution, so when used to demonstrate an experiment post analysis
  there will be no difference between control and treatment.

  Args:
    rng: The random number generator used to generate the data.
    n_items: The number of items in the data.
    impressions_average: The average impressions per item per day.
    impressions_standard_deviation: The standard deviation of impressions across
      items.
    ctr_average: The average CTR for the items.
    ctr_standard_deviation: The standard deviation of the CTR of the items.
    cpc_average: The average CPC for the items.
    cpc_item_level_standard_deviation: The standard deviation of the average CPC
      of the items.
    cpc_daily_level_standard_deviation: The standard deviation of the CPC per
      day within each item.
    conversion_rate_average: The average conversion rate of the items.
    conversion_rate_standard_deviation: The standard deviation of the conversion
      rates of the items.
    conversion_value_average: The average conversion value of the items.
    conversion_value_item_level_standard_deviation: The standard deviation of
      the average conversion value per item.
    conversion_value_daily_level_standard_deviation: The standard deviation of
      the conversion value per day for each item.
    weeks_before_start_date: The number of weeks of data to simulate before the
      start date.
    weeks_after_start_date: The number of weeks of data to simulate after the
      start date.
    start_date: The reference date for the experiment.

  Returns:
    Dataframe with synthetic data including the columns date, item_id,
    impressions, clicks, conversions, total_cost, total_conversion_value.

  Raises:
    ValueError: If the parameters for the simulation are outside the expected
      range for their distribution. All the ratios (CTR, CPC and conversion_rate
      must be between 0 and 1, and all the others must be greater than 0.)
  """

  date_fmt = "%Y-%m-%d"
  start_date_dt = dt.datetime.strptime(start_date, date_fmt)
  dates_after_start_date = [
      (start_date_dt + dt.timedelta(days=i)).strftime(date_fmt)
      for i in range(weeks_after_start_date * 7)
  ]
  dates_before_start_date = [
      (start_date_dt - dt.timedelta(days=i + 1)).strftime(date_fmt)
      for i in range(weeks_before_start_date * 7)
  ]
  dates = dates_after_start_date + dates_before_start_date

  item_ids = [f"item_{i}" for i in range(n_items)]

  dates_matrix, item_ids_matrix = np.meshgrid(dates, item_ids)
  shape = dates_matrix.shape

  # Expected impressions per day come from a log normal dist
  expected_impressions = _sample_from_lognormal(
      rng,
      mean=impressions_average,
      standard_deviation=impressions_standard_deviation,
      size=(n_items, 1),
  )

  # Expected CTR comes from a beta distribution
  expected_ctr = _sample_from_beta(
      rng,
      mean=ctr_average,
      standard_deviation=ctr_standard_deviation,
      size=(n_items, 1),
  )

  # Expected CPC per item comes from a log normal distribution
  expected_cpc = _sample_from_lognormal(
      rng,
      mean=cpc_average,
      standard_deviation=cpc_item_level_standard_deviation,
      size=(n_items, 1),
  )

  # Expected conversion rate comes from a beta distribution
  expected_conversion_rate = _sample_from_beta(
      rng,
      mean=conversion_rate_average,
      standard_deviation=conversion_rate_standard_deviation,
      size=(n_items, 1),
  )

  # Expected conversion value comes from a log normal distribution
  expected_conversion_value = _sample_from_lognormal(
      rng,
      mean=conversion_value_average,
      standard_deviation=conversion_value_item_level_standard_deviation,
      size=(n_items, 1),
  )

  # Now we sample the daily values, by broadcasting, so each of the
  # "expected_*" values above are constant for each item, but the values
  # we sample now are different for each item+day combination.
  impressions = rng.poisson(lam=expected_impressions, size=shape)
  clicks = rng.binomial(n=impressions, p=expected_ctr)
  cpc = _sample_from_lognormal(
      rng,
      mean=expected_cpc,
      standard_deviation=cpc_daily_level_standard_deviation,
      size=shape,
  )
  conversions = rng.binomial(n=clicks, p=expected_conversion_rate)
  average_conversion_value = _sample_from_lognormal(
      rng,
      mean=expected_conversion_value,
      standard_deviation=conversion_value_daily_level_standard_deviation,
      size=shape,
  )
  total_cost = cpc * clicks
  total_conversion_value = average_conversion_value * conversions

  data = pd.DataFrame({
      "date": dates_matrix.flatten(),
      "item_id": item_ids_matrix.flatten(),
      "impressions": impressions.flatten(),
      "clicks": clicks.flatten(),
      "conversions": conversions.flatten(),
      "total_cost": total_cost.flatten(),
      "total_conversion_value": total_conversion_value.flatten(),
  })

  return data


def standardize_column_names_and_types(
    data: pd.DataFrame,
    *,
    item_id_column: str,
    date_column: str,
    clicks_column: str | None = None,
    impressions_column: str | None = None,
    total_cost_column: str | None = None,
    conversions_column: str | None = None,
    total_conversion_value_column: str | None = None,
    custom_columns: dict[str, str] | None = None,
    custom_parse_functions: (
        dict[str, Callable[[pd.Series], pd.Series]] | None
    ) = None,
) -> pd.DataFrame:
  """Standardizes the column names and types.

  This renames the column names to their standardized names, casts the columns
  to a standardized data type, and drops any other columns. The item_id and
  date columns are requied, but all others are optional. However typically at
  least one other metric column is needed downstream.

  The custom_columns and custom_parse_functions are there to allow the user to
  include their own custom metrics, but they should not overlap the standard
  columns.

  Args:
    data: synthetic, historical, or runtime data
    item_id_column: The name of the column containing the unique identifier of
      the item in data.
    date_column: The name of the date column in data.
    clicks_column: The name of the clicks column in data if it exists, otherwise
      None. Defaults to None.
    impressions_column: The name of the impressions column in data if it exists,
      otherwise None. Defaults to None.
    total_cost_column: The name of the column containing the total cost (spend)
      in data if it exists, otherwise None. Defaults to None.
    conversions_column: The name of the conversions column in data if it exists,
      otherwise None. Defaults to None.
    total_conversion_value_column: The name of the column containing the total
      conversion value in data if it exists, otherwise None. Defaults to None.
    custom_columns: Key value pairs, where the key is the output column name and
      the value is the input column name, for any extra columns that have not
      already been specified.
    custom_parse_functions: Optionally, a dictionary containing functions to
      cast the custom columns to the correct data type. The keys are the
      standard column names, and the values are the functions which should take
      as an input a pandas series and return a pandas series.

  Returns:
    Dataframe with standardized column names and types.

  Raises:
    ValueError: If any of the custom columns overlap with the standard columns,
      or if any of the custom_parse_functions don't exist in the custom columns.
    ValueError: If the data does not contain the required columns.
  """
  custom_columns = custom_columns or {}
  custom_parse_functions = custom_parse_functions or {}

  parse_functions = {
      "item_id": lambda x: x.astype(str),
      "date": pd.to_datetime,
      "clicks": lambda x: pd.to_numeric(x).astype(int),
      "impressions": lambda x: pd.to_numeric(x).astype(int),
      "total_cost": lambda x: pd.to_numeric(x).astype(float),
      "conversions": lambda x: pd.to_numeric(x).astype(int),
      "total_conversion_value": lambda x: pd.to_numeric(x).astype(float),
  }
  column_names = {
      "item_id": item_id_column,
      "date": date_column,
      "clicks": clicks_column,
      "impressions": impressions_column,
      "total_cost": total_cost_column,
      "conversions": conversions_column,
      "total_conversion_value": total_conversion_value_column,
  }

  for custom_column in custom_columns.keys():
    if custom_column in column_names.keys():
      raise ValueError(
          f"The custom column {custom_column} is one of the standard columns,"
          f" set it with the {custom_column}_column argument."
      )

  for custom_parse_function in custom_parse_functions.keys():
    if custom_parse_function in parse_functions.keys():
      raise ValueError(
          f"The custom parse function {custom_parse_function} is one of the"
          " standard parse functions, no need to set it."
      )
    if custom_parse_function not in custom_columns.keys():
      raise ValueError(
          f"The custom parse function {custom_parse_function} must also be set"
          " in custom_columns."
      )

  parse_functions = parse_functions | custom_parse_functions
  column_names = column_names | custom_columns

  column_names = {
      parsed_name: input_name
      for parsed_name, input_name in column_names.items()
      if input_name is not None
  }
  parse_functions = {
      parsed_name: function
      for parsed_name, function in parse_functions.items()
      if parsed_name in column_names.keys()
  }
  column_renamer = {
      input_name: parsed_name
      for parsed_name, input_name in column_names.items()
  }

  for input_column in column_names.values():
    if input_column not in data.columns:
      raise ValueError(
          f"The input column {input_column} does not exist in the data."
      )

  output_data = data[list(column_names.values())].copy()
  output_data.rename(column_renamer, axis=1, inplace=True)

  for column in output_data.columns:
    parse_function = parse_functions.get(column, lambda x: x)
    output_data[column] = parse_function(output_data[column])

  return output_data


def fill_missing_rows_with_zeros(
    data: pd.DataFrame,
    item_id_column: str = "item_id",
    date_column: str = "date",
) -> pd.DataFrame:
  """Adds any missing rows in the dataframe with zeros.

  Often the input data is missing the rows where the item got 0 for all the
  metrics. However, this will break some of the analysis functions, so this
  function makes sure that there is at least one row for every unique
  combination of item_id and date.

  Args:
    data: synthetic, historical, or runtime data
    item_id_column: The column name containing the item id. Defaults to
      "item_id".
    date_column: The column name containing the date. Defaults to "date".

  Returns:
    Data columns in specified data type with missing date / item_id combinations
    filled with zeros.

  Raises:
    ValueError: If the item_id_column or date_column do not exist in the data.
  """
  required_columns = {item_id_column, date_column}
  missing_columns = required_columns - set(data.columns.values)
  if missing_columns:
    raise ValueError(
        "The data is missing the following required columns: "
        f"{missing_columns}."
    )

  all_items = data[[item_id_column]].drop_duplicates()
  all_dates = data[[date_column]].drop_duplicates()
  all_item_date_combinations = all_items.merge(all_dates, how="cross")
  output_data = all_item_date_combinations.merge(
      data, on=[item_id_column, date_column], how="left"
  )

  metric_columns = [
      column
      for column in output_data.columns
      if column not in [date_column, item_id_column]
  ]
  for metric_column in metric_columns:
    original_dtype = data[metric_column].dtype
    output_data[metric_column] = (
        output_data[metric_column].fillna(0).astype(original_dtype)
    )

  return output_data


def downsample_items(
    data: pd.DataFrame,
    downsample_fraction: float,
    item_id_column: str,
    rng: np.random.RandomState,
) -> pd.DataFrame:
  """Downsamples the items in the data.

  This will sample downsample_fraction of the items in the original data without
  replacement. If downsample_fraction = 1, it will return the original data
  unchanged.

  Args:
    data: The data to be downsampled.
    downsample_fraction: The fraction of items to select from the data, must be
      between 0 and 1. If 1, all items are returned.
    item_id_column: The column in the data containing the item identifier.
    rng: The random state used to ensure sampling is reproducable.

  Returns:
    Downsampled data.

  Raises:
    ValueError: If the downsample fraction is not in the range (0.0, 1.0], or
      if the item_id_column is not in the data.
  """
  if (downsample_fraction <= 0.0) | (downsample_fraction > 1.0):
    raise ValueError("Downsample_fraction must be in the range of (0.0, 1.0].")

  if item_id_column not in data.columns:
    raise ValueError(f"The column {item_id_column} does not exist in the data.")

  print(
      f"Data has {len(data.index.values)} rows and"
      f" {data[item_id_column].nunique()} items."
  )

  if downsample_fraction == 1.0:
    print("Not downsampling, since the downsample_fraction = 1.0")
    return data

  print(
      f"Sampling {downsample_fraction:.2%} of the items in the original data."
  )
  sampled_items = (
      data[item_id_column]
      .drop_duplicates()
      .sample(frac=downsample_fraction, random_state=rng)
  )
  data = data[data[item_id_column].isin(sampled_items)]
  print(
      f"Downsampled data has {len(data.index.values)} rows and"
      f" {data[item_id_column].nunique()} items."
  )
  return data


def add_week_id_and_week_start(
    data: pd.DataFrame,
    date_column: str,
    experiment_start_date: str | None = None,
    week_id_column: str = "week_id",
    week_start_column: str = "week_start",
) -> pd.DataFrame:
  """Returns the dataframe with the week_id and week start columns added.

  Args:
    data: The data to add the week id column to.
    date_column: The column name containing the dates in the data.
    experiment_start_date: The experiment start date to align the weeks to. Week
      ID 0 will start on the experiment start date. Must have the format
      YYYY-MM-DD. If not set, it will default to the first date in the data.
    week_id_column: The output column name that will contain the week ids.
      Defaults to "week_id".
    week_start_column: The name to give the output column that contains the date
      the week starts on. Defaults to "week_start".
  """
  if experiment_start_date is None:
    experiment_start_date_dt = data[date_column].min()
  else:
    experiment_start_date_dt = dt.datetime.strptime(
        experiment_start_date, "%Y-%m-%d"
    )

  days_offset = (data[date_column] - experiment_start_date_dt).dt.days
  data[week_id_column] = days_offset // 7
  data[week_start_column] = data[week_id_column].apply(
      lambda week_number: experiment_start_date_dt
      + dt.timedelta(days=week_number * 7)
  )
  return data


def group_data_to_complete_weeks(
    data: pd.DataFrame,
    date_column: dt.datetime,
    item_id_column: str,
    week_id_column: str,
    week_start_column: str,
    extra_group_columns: list[str] | None = None,
) -> pd.DataFrame:
  """Group input data to weekly data and drop any incomplete weeks.

  Daily and weekly data can be ingested. If the data provided is daily, this
  function will group by week, and if there are any incomplete weeks (weeks
  without 7 days), they will be removed. If weekly data is provided, no grouping
  will be done.

  Args:
    data: The data to be grouped to weekly
    date_column: The name of the column containing the date in the input data
    item_id_column: The name of the column containing the item_id in the input
      data
    week_id_column: The name of the column that contains the week_id.
    week_start_column: The name of the column that contains the week_start.
    extra_group_columns: Extra columns to include in the grouping. Must already
      be at weekly level.

  Returns:
    Data grouped by week.

  Raises:
    ValueError: If the input data is not weekly or daily.
  """
  extra_group_columns = extra_group_columns or []
  data_copy = data.copy()
  date_sorted_unique = data_copy[date_column].drop_duplicates().sort_values()
  frequency = (
      (date_sorted_unique - date_sorted_unique.shift(1)).iloc[1:].dt.days.values
  )

  if np.all(frequency == 1):
    print("The input data is daily, grouping to weekly.")
    non_aggregate_columns = [
        week_id_column,
        week_start_column,
        item_id_column,
        date_column,
    ] + extra_group_columns
    aggregations = {
        c: (c, "sum")
        for c in data_copy.columns.values
        if c not in non_aggregate_columns
    }
    aggregations["days_in_week"] = (date_column, "nunique")

    data_copy = (
        data_copy.groupby(
            [week_id_column, week_start_column, item_id_column]
            + extra_group_columns
        )
        .agg(**aggregations)
        .reset_index()
    )

    data_copy = (
        data_copy[data_copy["days_in_week"] == 7]
        .copy()
        .drop(columns=["days_in_week"])
    )
  elif np.all(frequency == 7):
    print("The input data is already weekly, grouping not required.")
    data_copy = data_copy.drop(columns=[date_column])
  else:
    raise ValueError(
        "The data is not daily or weekly. Cannot proceed with mixed frequency"
    )

  return data_copy


def _validate_every_value_exists_exactly_once_for_every_group(
    data: pd.DataFrame,
    *,
    value_column: str,
    group_column: str,
) -> None:
  """Validates that every value exists exactly once in every group."""
  value_counts_per_group = (
      data[[group_column, value_column]]
      .groupby(group_column)
      .count()[value_column]
  )

  unique_value_counts_per_group = (
      data[[group_column, value_column]]
      .groupby(group_column)
      .nunique()[value_column]
  )

  if np.any(value_counts_per_group != unique_value_counts_per_group):
    bad_groups = value_counts_per_group.loc[
        value_counts_per_group != unique_value_counts_per_group
    ].index.values
    raise ValueError(
        f"There are duplicate {value_column} when {group_column} in "
        f"{bad_groups}"
    )

  value_counts_first_group = value_counts_per_group.values[0]
  if np.any(value_counts_per_group.values != value_counts_first_group):
    raise ValueError(
        f"Some {group_column} are missing {value_column}. Below are the number "
        f"of unique {value_column} per {group_column}, which should be equal "
        f"for all dates.\n{value_counts_per_group}"
    )
  else:
    print(
        f"All {group_column} have {value_counts_first_group:,} {value_column},"
        " check passed."
    )


def _validate_no_null_values(data: pd.DataFrame) -> None:
  """Validates that there are no null or n/a values in the data."""
  nulls = data.isnull().sum()
  if np.any(nulls.values > 0):
    raise ValueError(f"Nulls found in data:\n{nulls}")

  nas = data.isna().sum()
  if np.any(nas.values > 0):
    raise ValueError(f"N/As found in data:\n{nas}")

  print("No nulls check passed.")


def _validate_dates_are_either_daily_or_weekly(
    data: pd.DataFrame, date_column: str, date_id_column: str
) -> None:
  """Validates that dates are daily or weekly spaced when sorted by date_id."""
  if not pd.api.types.is_datetime64_any_dtype(data[date_column]):
    raise ValueError(
        f"Date column must be datetime type, got {data[date_column].dtype}."
    )

  unique_dates = data.sort_values(date_id_column)[date_column].drop_duplicates()
  days_between_dates = (
      (unique_dates - unique_dates.shift(1)).iloc[1:].dt.days.values
  )

  if np.all(days_between_dates == 7):
    print("Dates are weekly, check passed.")
  elif np.all(days_between_dates == 1):
    print("Dates are daily, check passed.")
  else:
    raise ValueError(
        "Dates and neither consistently weekly spaced or consistently daily"
        " spaced, check failed."
    )


def _validate_date_ids_are_consecutive_integers(
    data: pd.DataFrame, date_id_column: str
) -> None:
  """Validates that the date_id is an array on consecutive integers."""
  if not pd.api.types.is_integer_dtype(data[date_id_column]):
    raise ValueError(
        f"Date_id column must be an integer, got {data[date_id_column].dtype}."
    )

  unique_date_ids = data[date_id_column].drop_duplicates().sort_values()
  gap_between_date_ids = (
      (unique_date_ids - unique_date_ids.shift(1)).iloc[1:].values
  )
  if not np.all(gap_between_date_ids == 1):
    raise ValueError("Date ids are not consecituve, there is a gap.")

  print("Date ids are consecutive integers, check passed.")


def _validate_all_metrics_are_finite(
    data: pd.DataFrame, metric_columns: Collection[str]
) -> None:
  """Validates that all the metrics are finite."""
  for metric_column in metric_columns:
    invalid_metric_count = (~np.isfinite(data[metric_column].values)).sum()

    if invalid_metric_count:
      raise ValueError(
          f"There are {invalid_metric_count} non-finite values in"
          f" {metric_column}. "
      )

  print(
      f"The following metric values are all finite: {metric_columns}, check"
      " passed."
  )


def _validate_all_metrics_are_positive(
    data: pd.DataFrame, metric_columns: Collection[str], required: bool
) -> None:
  """Validates that all the metrics are positive."""
  for metric_column in metric_columns:
    invalid_metric_count = (data[metric_column] < 0).sum()

    if invalid_metric_count:
      if required:
        raise ValueError(
            f"There are {invalid_metric_count} negative values in"
            f" {metric_column}. "
        )
      else:
        print(
            f"WARNING: There are {invalid_metric_count} negative values in"
            f" {metric_column}. This will make it difficult to interpret "
            "relative lift estimates."
        )

  if required:
    print(
        "The following metric values are all positive:"
        f" {metric_columns}, check passed."
    )


def validate_historical_data(
    historical_data: pd.DataFrame,
    item_id_column: str,
    date_column: str,
    date_id_column: str,
    primary_metric_column: str,
    require_positive_primary_metric: bool = True,
) -> None:
  """Runs all the required validation for the historical data.

  The historical data is used to perform simulations to design the optimal
  experiment. This validates that:

  - There are no null or n/a values in the data.
  - All items have exactly 1 row for every date, there are no missing
    date / item combinations or duplicates.
  - The dates are either daily or weekly.
  - All primary metric is finite. This is stricter than non-null as it ensures
    they are numeric and not infinite.
  - The date_id must be integers and consecutive.

  If require_positive_primary_metric is true, it also validates that the primary
  metric is always positive or 0.

  Args:
    historical_data: The historical data to be validated.
    item_id_column: The column in the data contining the item identifier.
    date_column: The column in the data containing the date. This column must
      have a datetime type.
    date_id_column: The column containing an integer identifier for the dates.
    primary_metric_column: The column containing the primary metric.
    require_positive_primary_metric: Require that the primary metric is always
      positive or 0. If set to False, it will check but not raise an exception
      if there are negative metrics. Defaults to True.

  Raises:
    ValueError: If any of the validations fail.
  """

  _validate_no_null_values(historical_data)
  _validate_every_value_exists_exactly_once_for_every_group(
      historical_data,
      group_column=date_column,
      value_column=item_id_column,
  )
  _validate_dates_are_either_daily_or_weekly(
      historical_data, date_column=date_column, date_id_column=date_id_column
  )
  _validate_date_ids_are_consecutive_integers(historical_data, date_id_column)
  _validate_all_metrics_are_finite(
      historical_data, metric_columns=[primary_metric_column]
  )
  _validate_all_metrics_are_positive(
      historical_data,
      metric_columns=[primary_metric_column],
      required=require_positive_primary_metric,
  )


def add_at_least_one_metrics(
    data: pd.DataFrame, metrics: dict[str, str]
) -> pd.DataFrame:
  """Returns the data with the "at least one" metrics added.

  The at least one metrics are binomial metrics, which are 1 when the metric
  is greater than 0, and 0 otherwise. They are useful, for example, for testing
  the impact on zombie items that typically get very few clicks or impressions.

  It will print a warning if the at least one metric is constant (all 0s or all
  1s). In this case you should not use that metric - it won't work in the
  analysis.

  Args:
    data: The data to add the "at least one metrics" to.
    metrics: A mapping between the original column name and the new column name
      for the "at least one" metric. For example, to calculate
      at_least_one_click from clicks you would set metrics = {"clicks":
      "at_least_one_click"}.
  """
  for metric, at_least_one_metric in metrics.items():
    if np.all(data[metric] > 0) | np.all(data[metric] <= 0):
      print(
          f"WARNING: {at_least_one_metric} is constant, it won't be useful to"
          " analyse."
      )
    data[at_least_one_metric] = (data[metric] > 0).astype(int)
  return data
