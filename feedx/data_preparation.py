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
import pandas as pd

GOOGLE_ADS_PERFORMANCE_CSV_COLUMNS = {
    "date_column": "Week",
    "item_id_column": "Item ID",
    "impressions_column": "Impr.",
    "clicks_column": "Clicks",
}
GOOGLE_ADS_PERFORMANCE_READ_CSV_ARGS = {"header": 2, "thousands": ","}


def generate_historical_synthetic_data(
    rng: np.random.Generator,
    n_items: int = 7000,
    impressions_average: float = 100.0,
    impressions_standard_deviation: float = 100.0,
    ctr_average: float = 0.02,
    ctr_standard_deviation: float = 0.015,
    historical_days: int = 90,
) -> pd.DataFrame:
  """Generates historical synthetic experiment data.

  This generates daily clicks and impressions data which can be used to
  demonstrate how FeedX designs experiments. It is simulated historical
  data, meaning data from before the start of the experiment.

  It uses a lognormal distribution for the expected impressions, since it is >=0
  and has
  a positive skew, mirroring many real world datasets. Each item has the same
  expected
  impressions for every date. The actual impressions are then sampled using a
  poisson distribution, with
  the mean being the expected impressions.

  The click through rate (CTR) is sampled for each item from a beta
  distribution,
  which ensures it is between 0 and 1. Then the number of clicks is sampled
  using a binomial distribution with the CTR and the number of impressions.

  Args:
    rng: The random number generator used to generate the data.
    n_items: The number of items in the data.
    impressions_average: The average impressions per item per day.
    impressions_standard_deviation: The standard deviation of impressions across
      items.
    ctr_average: The average CTR for the items.
    ctr_standard_deviation: The standard deviation of the CTR of the items.
    historical_days: The number of days of data to simulate.

  Returns:
    Dataframe with synthetic historical data.

  Raises:
    ValueError: If ctr_average or ctr_standard_deviation are less than 0 or
      greater than 1, or if the impressions_average or
      impressions_standard_devation are less than or equal to 0.
  """
  valid_ctr_average = ctr_average > 0 and ctr_average < 1
  valid_ctr_standard_deviation = (
      ctr_standard_deviation > 0 and ctr_standard_deviation < 1
  )
  valid_impressions_average = impressions_average > 0
  valid_impressions_standard_deviation = impressions_standard_deviation > 0

  if not valid_ctr_average:
    raise ValueError("ctr_average must be between 0 and 1")
  if not valid_ctr_standard_deviation:
    raise ValueError("ctr_standard_deviation must be between 0 and 1")
  if not valid_impressions_average:
    raise ValueError("impressions_average must be greater than 0")
  if not valid_impressions_standard_deviation:
    raise ValueError("impressions_standard_deviation must be greater than 0")

  date_fmt = "%Y-%m-%d"
  today = dt.date.today()
  dates = [
      (today - dt.timedelta(days=i + 1)).strftime(date_fmt)
      for i in range(historical_days)
  ]
  item_ids = [f"item_{i}" for i in range(n_items)]

  dates_matrix, item_ids_matrix = np.meshgrid(dates, item_ids)

  shape = dates_matrix.shape

  mean = np.log(impressions_average) - 0.5 * np.log(
      impressions_standard_deviation**2 / impressions_average**2 + 1
  )
  sigma = np.sqrt(
      np.log(impressions_standard_deviation**2 / impressions_average**2 + 1)
  )
  expected_impressions = rng.lognormal(
      mean=mean, sigma=sigma, size=n_items
  ).reshape(-1, 1)

  beta_dist_params = (
      np.sqrt(ctr_average * (1.0 - ctr_average)) / ctr_standard_deviation
  )
  expected_ctr = rng.beta(
      a=ctr_average * beta_dist_params,
      b=(1.0 - ctr_average) * beta_dist_params,
      size=n_items,
  ).reshape(-1, 1)

  impressions = rng.poisson(lam=expected_impressions, size=shape)

  clicks = rng.binomial(n=impressions, p=expected_ctr)

  data = pd.DataFrame({
      "date": dates_matrix.flatten(),
      "item_id": item_ids_matrix.flatten(),
      "impressions": impressions.flatten(),
      "clicks": clicks.flatten(),
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
