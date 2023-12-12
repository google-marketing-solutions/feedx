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
from scipy import stats

from feedx import experiment_design

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


def add_relative_effects_to_synthetic_data_for_demo(
    synthetic_data: pd.DataFrame,
    *,
    time_period_column: str,
    treatment_assignment_column: str,
    impressions_relative_effect: float = 0.1,
    clicks_relative_effect: float = 0.2,
    conversions_relative_effect: float = 0.2,
    total_conversion_value_relative_effect: float = 0.2,
    total_cost_relative_effect: float = 0.2,
) -> pd.DataFrame:
  """Adds relative effects to synthetic data for the demo notebook.

  DO NOT USE THIS FOR ANY REAL EXPERIMENT ANALYSIS.

  This will add a synthetic relative impact for the items when they are treated.
  It will infer based on the values in the time_period_column whether it is a
  crossover or regular experiment. For a crossover experiment it will add
  the effect also in the washout periods.

  Since this adds a relative effect, it will leave all the zeros as zeros. This
  means any binomial metrics like "at_least_one_click" wont be affected.

  Args:
    synthetic_data: The synthetic data to add the effects to, generated by
      generate_synthetic_data().
    time_period_column: The column in the data containing the time period.
    treatment_assignment_column: The column identifying whether the items are in
      control (0) or treatment (1).
    impressions_relative_effect: The relative effect to apply to the
      impressions, defaults to 0.1.
    clicks_relative_effect: The relative effect to apply to the clicks, defaults
      to 0.2.
    conversions_relative_effect: The relative effect to apply to the
      conversions, defaults to 0.2.
    total_conversion_value_relative_effect: The relative effect to apply to the
      total_conversion_value, defaults to 0.2.
    total_cost_relative_effect: The relative effect to apply to the total_cost,
      defaults to 0.2.
  """
  synthetic_data = synthetic_data.copy()
  is_crossover = "test_1" in synthetic_data[time_period_column].values

  if is_crossover:
    treated_period_1_mask = synthetic_data[time_period_column].isin(
        ["test_1", "washout_1"]
    ) & (synthetic_data[treatment_assignment_column] == 1)
    treated_period_2_mask = synthetic_data[time_period_column].isin(
        ["test_2", "washout_2"]
    ) & (synthetic_data[treatment_assignment_column] == 0)
    treated_mask = treated_period_1_mask | treated_period_2_mask
  else:
    treated_mask = (synthetic_data[time_period_column] == "test") & (
        synthetic_data[treatment_assignment_column] == 1
    )

  synthetic_data.loc[treated_mask, "impressions"] *= (
      1.0 + impressions_relative_effect
  )
  synthetic_data.loc[treated_mask, "clicks"] *= 1.0 + clicks_relative_effect
  synthetic_data.loc[treated_mask, "conversions"] *= (
      1.0 + conversions_relative_effect
  )
  synthetic_data.loc[treated_mask, "total_conversion_value"] *= (
      1.0 + total_conversion_value_relative_effect
  )
  synthetic_data.loc[treated_mask, "total_cost"] *= (
      1.0 + total_cost_relative_effect
  )

  return synthetic_data


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


def _validate_number_of_items_matches_design(
    experiment_data: pd.DataFrame,
    design: experiment_design.ExperimentDesign,
    item_id_column: str,
) -> None:
  """Validates the number of items matches n_items_after_pre_trim in design."""
  n_items = experiment_data[item_id_column].nunique()
  if n_items != design.n_items_after_pre_trim:
    raise ValueError(
        f"The experiment data has {n_items} unique item_ids, but"
        f" {design.n_items_after_pre_trim = }."
    )

  print("Number of items matches design, check passed.")


def _validate_experiment_dates_match(
    experiment_data: pd.DataFrame,
    design: experiment_design.ExperimentDesign,
    date_column: str,
    experiment_start_date: str,
    experiment_has_concluded: bool,
) -> None:
  """Validates that the experiment dates match.

  This checks that the data contains the earliest and latest dates required,
  based on the runtime and pretest weeks in the design and the experiment start
  date.
  """
  start_date_dt = dt.datetime.strptime(experiment_start_date, "%Y-%m-%d")
  # first day of first pretest week
  expected_min_date = start_date_dt - dt.timedelta(
      days=design.pretest_weeks * 7
  )
  # first day of final runtime week
  expected_max_date = start_date_dt + dt.timedelta(
      days=(design.runtime_weeks - 1) * 7
  )

  actual_min_date = experiment_data[date_column].min()
  actual_max_date = experiment_data[date_column].max()

  if actual_min_date > expected_min_date:
    raise ValueError(
        "Experiment data does not go back far enough. Design expects"
        f" {design.pretest_weeks} weeks of pretest data, and experiment starts"
        f" on {experiment_start_date}, so the first date must be"
        f" {expected_min_date} or earlier, but the earliest date is"
        f" {actual_min_date}"
    )

  if experiment_has_concluded:
    if actual_max_date < expected_max_date:
      raise ValueError(
          "The experiment has concluded but you are missing data from the end"
          f" of the experiment. Design expects {design.runtime_weeks} weeks of"
          f" runtime data, and experiment starts on {experiment_start_date}, so"
          f" the final week must be {expected_max_date} or later, but latest"
          f" date is {actual_max_date}."
      )

  print("Date range matches design, check passed.")


def _validate_treatment_assignment_is_unique_for_each_item_id(
    experiment_data: pd.DataFrame,
    item_id_column: str,
    treatment_assignment_column: str,
) -> None:
  n_assignments = experiment_data.groupby(item_id_column)[
      treatment_assignment_column
  ].nunique()
  if np.any(n_assignments != 1):
    raise ValueError(
        "Some items have non-unique treatment assignments:"
        f" {n_assignments[n_assignments != 1]}"
    )

  print("Treatment assignment is unique for each item, check passed.")


def _validate_coinflip_matches_assignments(
    experiment_data: pd.DataFrame,
    design: experiment_design.ExperimentDesign,
    treatment_assignment_column: str,
    item_id_column: str,
) -> None:
  actual_assignments = experiment_data[treatment_assignment_column].values
  coinflip = experiment_design.Coinflip(salt=design.coinflip_salt)
  expected_assignments = experiment_data[item_id_column].apply(coinflip).values

  if not np.array_equal(actual_assignments, expected_assignments):
    fraction_wrong = np.mean(actual_assignments != expected_assignments)
    number_wrong = np.sum(actual_assignments != expected_assignments)
    raise ValueError(
        f"{number_wrong} ({fraction_wrong:.2%}) of the treatment assigments in"
        " the data do not match what was expected from the coinflip salt in"
        " the design. This could mean you have loaded the wrong design or"
        " treatment assignment files."
    )

  print("Coinflip salt from design matches assignments, check passed.")


def validate_no_sample_ratio_mismatch(
    experiment_data: pd.DataFrame,
    item_id_column: str,
    treatment_assignment_column: str,
    p_value_threshold: float = 0.001,
) -> None:
  """Validates there is no sample ratio mismatch in the treatment assignment.

  Sample ratio mismatch (SRM) occures when the number of samples in control and
  treatment does not match the expected split (in this case 50/50). This
  checks this by estimating a p-value and if the p-value is smaller than the
  p_value_threshold it raises an error. The p value threshold should be very
  small, otherwise this validation will fail very often just by chance, and
  if there is a real SRM problem it should be quite obvious.

  Args:
    experiment_data: The data to run the validation on.
    item_id_column: The column containing the item id.
    treatment_assignment_column: The column containing the treatment assignment,
      0 for control and 1 for treatment.
     p_value_threshold: The threshold to compare the p-value against. Defaults
       to 0.001.

  Raises:
    If the p-value for the null hypothesis that the probability of being
    assigned to control is 50% is smaller than the p-value threshold.
  """
  counts = experiment_data.groupby(treatment_assignment_column)[
      item_id_column
  ].nunique()
  srm_result = stats.binomtest(
      counts[1], n=counts[1] + counts[0], p=0.5, alternative="two-sided"
  )

  if srm_result.pvalue < p_value_threshold:
    raise ValueError(f"Sample ratio mismatch detected! {srm_result}")

  print("No Sample Ratio Mismatch, check passed.")


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


def validate_experiment_data(
    experiment_data: pd.DataFrame,
    *,
    design: experiment_design.ExperimentDesign,
    item_id_column: str,
    date_column: str,
    date_id_column: str,
    treatment_assignment_column: str,
    metric_columns: Collection[str],
    experiment_start_date: str,
    experiment_has_concluded: bool = True,
    can_be_negative_metric_columns: Collection[str] | None = None,
) -> None:
  """Runs all the required validation for the experiment data.

  This validates that:

  - There are no null or n/a values in the data.
  - All items have exactly 1 row for every date, there are no missing
    date / item combinations or duplicates.
  - The dates are either daily or weekly.
  - All metrics are finite. This is stricter than non-null as it ensures
    they are numeric and not infinite.
  - The date_id must be integers and consecutive.
  - The metrics are not negative, unless they are in
    can_be_negative_metric_columns.
  - The number of items in the data matches the expected number in the design.
  - The earliest and last date in the data match what is required for the design
    given the experiment start date and the runtime and pretest weeks. If
    experiment_has_concluded is set to False, it only checks the earliest date.
  - The primary metric exists as a column in the data.
  - Check that the treatment assignments are unique for each item and they match
    what is expected from the coinflip salt. If they don't match the salt,
    it means the design or treatment assignments dont match.

  Args:
    experiment_data: The experiment data to be validated.
    design: The experiment design for this experiment.
    item_id_column: The column in the data contining the item identifier.
    date_column: The column in the data containing the date. This column must
      have a datetime type.
    date_id_column: The column containing an integer identifier for the dates.
    treatment_assignment_column: The column containing the treatment assignment,
      0 if the item is in the control group and 1 if it is in the treatment
      group.
    metric_columns: The columns containing the metrics to be analyzed. Must
      include the primary metric from the design.
    experiment_start_date: The date the experiment started. Must have the format
      YYYY-MM-DD.
    experiment_has_concluded: Has the experiment finished (reached it's planned
      runtime)? Defaults to True.
    can_be_negative_metric_columns: The list of metric columns that are allowed
      to be negative. Defaults to None, meaning all metrics must be positive.

  Raises:
    ValueError: If any of the validations fail.
  """
  if design.primary_metric not in metric_columns:
    raise ValueError(
        f"The primary metric {design.primary_metric} must be one of the"
        " metric_columns."
    )

  _validate_no_null_values(experiment_data)
  _validate_every_value_exists_exactly_once_for_every_group(
      experiment_data,
      group_column=date_column,
      value_column=item_id_column,
  )
  _validate_dates_are_either_daily_or_weekly(
      experiment_data, date_column=date_column, date_id_column=date_id_column
  )
  _validate_date_ids_are_consecutive_integers(experiment_data, date_id_column)
  _validate_all_metrics_are_finite(
      experiment_data, metric_columns=metric_columns
  )

  can_be_negative_metric_columns = can_be_negative_metric_columns or []
  require_non_negative_metric_columns = [
      metric_column
      for metric_column in metric_columns
      if metric_column not in can_be_negative_metric_columns
  ]
  if require_non_negative_metric_columns:
    _validate_all_metrics_are_positive(
        experiment_data,
        metric_columns=require_non_negative_metric_columns,
        required=True,
    )
  if can_be_negative_metric_columns:
    _validate_all_metrics_are_positive(
        experiment_data,
        metric_columns=can_be_negative_metric_columns,
        required=False,
    )

  _validate_number_of_items_matches_design(
      experiment_data, design, item_id_column
  )
  _validate_experiment_dates_match(
      experiment_data,
      design=design,
      date_column=date_column,
      experiment_start_date=experiment_start_date,
      experiment_has_concluded=experiment_has_concluded,
  )

  _validate_treatment_assignment_is_unique_for_each_item_id(
      experiment_data, item_id_column, treatment_assignment_column
  )
  _validate_coinflip_matches_assignments(
      experiment_data, design, treatment_assignment_column, item_id_column
  )
  validate_no_sample_ratio_mismatch(
      experiment_data, item_id_column, treatment_assignment_column
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


def prepare_and_validate_historical_data(
    raw_data: pd.DataFrame,
    *,
    item_id_column: str,
    date_column: str,
    primary_metric: str,
    primary_metric_column: str,
    rng: np.random.RandomState,
    item_sample_fraction: float = 1.0,
    require_positive_primary_metric: bool = True,
) -> pd.DataFrame:
  """Prepares and validates the historical data for the experiment design.

  This performs the following steps before returning the cleaned data:

    1. Standardize the column names and types.
    2. Fill any missing date + item_id combinations with zeros.
    3. Add a week_id and week start date column, and group the data so that
       it's weekly if it's not already (weeks are aligned to start on the
       minimum date).
    4. Downsample items if required.
    5. Perform validation checks on the data.

  Args:
    raw_data: The raw data to be processed and validated.
    item_id_column: The name of the column containing the item ids.
    date_column: The name of the column containing the dates.
    primary_metric: The name of the primary metric that will be used. Must be
      one of clicks, impressions, conversions, total_conversion_value,
      total_cost or other.
    primary_metric_column: The column name containing the primary metric.
    rng: The random number generator used if downsampling is applied.
    item_sample_fraction: The faction of items to sample if doing downsampling.
      If set to 1, no downsampling is done. Defaults to 1.
    require_positive_primary_metric: Whether the primary metric should always be
      positive or not. Defaults to true.

  Returns:
    The processed data.
  """
  allowed_primary_metrics = [
      "clicks",
      "impressions",
      "conversions",
      "total_conversion_value",
      "total_cost",
      "other",
  ]
  if primary_metric not in allowed_primary_metrics:
    raise ValueError(
        f"The primary_metric must be one of {allowed_primary_metrics}."
    )

  standardize_column_names_and_types_args = {
      "date_column": date_column,
      "item_id_column": item_id_column,
  }

  if primary_metric == "other":
    standardize_column_names_and_types_args["custom_columns"] = {
        primary_metric_column: primary_metric_column
    }
    standardize_column_names_and_types_args["custom_parse_functions"] = {
        primary_metric_column: lambda x: x.astype(float)
    }
    primary_metric = primary_metric_column
  else:
    standardize_column_names_and_types_args[primary_metric + "_column"] = (
        primary_metric_column
    )

  clean_data = (
      raw_data.pipe(
          standardize_column_names_and_types,
          **standardize_column_names_and_types_args,
      )
      .pipe(fill_missing_rows_with_zeros)
      .pipe(add_week_id_and_week_start, date_column="date")
      .pipe(
          group_data_to_complete_weeks,
          date_column="date",
          item_id_column="item_id",
          week_id_column="week_id",
          week_start_column="week_start",
      )
      .pipe(
          downsample_items,
          downsample_fraction=item_sample_fraction,
          item_id_column="item_id",
          rng=rng,
      )
  )

  validate_historical_data(
      clean_data,
      item_id_column="item_id",
      date_column="week_start",
      date_id_column="week_id",
      primary_metric_column=primary_metric,
      require_positive_primary_metric=require_positive_primary_metric,
  )

  return clean_data


def trim_outliers(
    data: pd.DataFrame,
    order_by: str | tuple[str, ...],
    trim_percentile_top: float,
    trim_percentile_bottom: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
  """Trims outliers from the data.

  1. Calculates the number of rows to trim from the top and bottom by
    multiplying the trim percentile by the number of rows in the data, and
    flooring the result.
  2. If 0 rows are to be trimmed, return the data.
  3. Rank the data by the order_by column, with ties being given a random order
    using the random number generator rng.
  4. Remove the rows to be trimmed from either end and return the data.

  Args:
    data: The data to be trimmed.
    order_by: The column to order by to select the outliers to trim.
    trim_percentile_top: The fraction of the highest values to remove from the
      data.
    trim_percentile_bottom: The fraction of the lowest values to remove from the
      data.
    rng: The random number generator used to break ties in the ordering.

  Returns:
    The data with the outliers trimmed.

  Raises:
    ValueError: If either of the trim_perentile_top or trim_percentile_bottom
      are negative, or if trim_perentile_top + trim_percentile_bottom >= 1.0.
      This is to ensure that you do not trim 100% of the data.
  """
  if trim_percentile_top < 0.0:
    raise ValueError("trim_percentile_top must be >= 0.0.")
  if trim_percentile_bottom < 0.0:
    raise ValueError("trim_percentile_bottom must be >= 0.0.")
  if trim_percentile_top + trim_percentile_bottom >= 1.0:
    raise ValueError(
        "trim_percentile_top + trim_percentile_bottom must be < 1.0, otherwise"
        f" you will remove all the data. {trim_percentile_top = }, "
        f"{trim_percentile_bottom = }, so "
        f"{trim_percentile_top + trim_percentile_bottom = } >= 1.0."
    )

  n_samples = len(data.index.values)
  n_trim_top = int(np.floor(n_samples * trim_percentile_top))
  n_trim_bottom = int(np.floor(n_samples * trim_percentile_bottom))

  if (n_trim_top == 0) & (n_trim_bottom == 0):
    return data

  min_rank = n_trim_top
  max_rank = n_samples - n_trim_bottom

  rank = data.sample(frac=1.0, random_state=rng)[order_by].rank(
      method="first", ascending=False
  )

  return data.loc[(rank > min_rank) & (rank <= max_rank)].copy()


def perform_treatment_assignment(
    historical_data: pd.DataFrame,
    *,
    design: experiment_design.ExperimentDesign,
    rng: np.random.Generator,
    item_id_column: str,
    week_id_column: str | None = None,
    treatment_assignment_column: str = "treatment_assignment",
) -> pd.DataFrame:
  """Returns the items for the experiment with their treatment assignment.

  This function performs the pre-test trimming if specified in the experiment
  design, using the most recent weeks in the historical data for the pre-test
  weeks.

  Then, with the final set of items, it randomises them into control (0)
  or treatment (1) using a coinflip. The salt from the coinflip is taken
  from the experiment design if it exists, and otherwise a random salt is
  generated and added to the design.

  Args:
    historical_data: The historical data to use to get the items and perform the
      trimming.
    design: The experiment design.
    rng: The random number generator, used to break ties in the trimming.
    item_id_column: The name of the column in historical_data containing the
      item identifier.
    week_id_column: The name of the column in historical_data containing the
      week identifier. This is only used to apply pre-test trimming, if the
      design does do pre-test trimming this can be set to None. Defaults to
      None.
    treatment_assignment_column: The name of the output column that will contain
      the treatment assignment. Defaults to "treatment_assignment".

  Returns:
    A dataframe containing the item_ids selected for the experiment (after
    pre-trimming), and the treatment assignment generated from the item id and
    coinflip salt.

  Raises:
    ValueError: If the item_id_column or week_id_column (if it's needed) are not
      in the historical_data, or if the treatment_assignment_column is the same
      as the item_id_column.
  """
  design_has_pre_trimming = (
      design.pre_trim_top_percentile > 0.0
      or design.pre_trim_bottom_percentile > 0.0
  )

  required_columns = {item_id_column}
  if design_has_pre_trimming & (week_id_column is not None):
    required_columns.add(week_id_column)
  elif design_has_pre_trimming:
    raise ValueError("Must specify week_id_column if design has pre-trimming.")

  missing_columns = required_columns - set(historical_data.columns)
  if missing_columns:
    raise ValueError(
        "The historical_data is missing the following required columns: "
        f"{missing_columns}"
    )

  if item_id_column == treatment_assignment_column:
    raise ValueError(
        "The treatment_assignment_column must not be the same as the"
        f" item_id_column, both are '{item_id_column}'."
    )

  if design.coinflip_salt:
    coinflip = experiment_design.Coinflip(design.coinflip_salt)
    print(f"Using coinflip salt from design: {coinflip.salt}")
  else:
    coinflip = experiment_design.Coinflip.with_random_salt()
    design.coinflip_salt = coinflip.salt
    print(f"Generating random coinflip salt: {coinflip.salt}")

  if design_has_pre_trimming:
    max_week_id = historical_data[week_id_column].max()
    start_week_id = max_week_id - design.pretest_weeks
    pretest_data = historical_data.loc[
        historical_data[week_id_column] > start_week_id
    ].copy()
    pretest_items = (
        pretest_data.groupby(item_id_column)[design.primary_metric]
        .sum()
        .reset_index()
    )

    experiment_items = trim_outliers(
        pretest_items,
        order_by=design.primary_metric,
        trim_percentile_bottom=design.pre_trim_bottom_percentile,
        trim_percentile_top=design.pre_trim_top_percentile,
        rng=rng,
    )[[item_id_column]]
  else:
    experiment_items = (
        historical_data[[item_id_column]].copy().drop_duplicates()
    )

  experiment_items[treatment_assignment_column] = experiment_items[
      item_id_column
  ].apply(coinflip)

  validate_no_sample_ratio_mismatch(
      experiment_items,
      item_id_column=item_id_column,
      treatment_assignment_column=treatment_assignment_column,
  )

  return experiment_items
