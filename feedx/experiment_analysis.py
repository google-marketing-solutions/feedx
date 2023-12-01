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

"""Module containing functions for analyzing experiment results."""

import dataclasses
from typing import Dict

import numpy as np
import pandas as pd

from feedx import experiment_design
from feedx import statistics

ExperimentDesign = experiment_design.ExperimentDesign


def get_weeks_between(
    week_number: pd.Series, start_id: int, end_id: int
) -> pd.Series:
  """Computes a boolean array indicating if the week number is between start_id

      and end_id (inclusive).

  This function will be used to assign treatment time to items based on the
  selected number of runtime weeks and regular vs crossover experiment.

  Args:
    week_number: week number associated with a specific date in the dataset.
    start_id: integer value representing the starting point of a test phase.
    end_id: integer value representing the ending point of a test phase.

  Returns:
    True or false if the week number is between start_id and end_id indicating a
      specific test phase. This result will assign the corresponding test phase
      to items.
  """
  greater = week_number >= start_id
  less = week_number <= end_id
  return greater & less


def trim_outliers(
    data: pd.DataFrame,
    order_by: str,
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

  return data.loc[(rank > min_rank) & (rank <= max_rank)]


def _pivot_regular_time_assignment(
    data: pd.DataFrame,
    *,
    start_week_id: int,
    pretest_weeks: int,
    runtime_weeks: int,
    metric_columns: list[str],
    item_id_column: str,
    week_id_column: str,
    treatment_column: str | None = None,
    time_period_column: str = "time_period",
) -> pd.DataFrame:
  """Pivots the data from a regular A/B test so that it's ready for analysis.

  This function pivots the input dataset so that each row is an independent
  randomisation unit, and the columns are combinations of (metric, time_period).
  If treatment_column is not specified then the index is just the item_id,
  while if it is specified then the index is pairs of (item_id, treatment).

  The time_period is either pretest (for the period before the test began), or
  test (for the period during the experiment). If there is no pretest period,
  then it will only contain the test period.

  The values are the average of the metric per week during the time_period.

  Args:
    data: The data from the experiment to be pivoted.
    start_week_id: The id of the first week of the experiment (inclusive).
    pretest_weeks: The number of weeks to use as pre-test weeks.
    runtime_weeks: The number of weeks to use as experiment runtime.
    metric_columns: The list of metrics to be analyzed.
    item_id_column: The column identifying individual randomisation units.
    week_id_column: The column identifying individual weeks.
    treatment_column: The column that indicates if the item_id is in the
      treatment or control group. None if there is no treatment column.
    time_period_column: The column name used internally to do the time
      assignment. It will be the name of the time_period index in the returned
      dataframe.

  Returns:
    The pivoted data, where rows are randomisation units, and columns are
    combinations of metrics and time_periods.
  """
  is_pretest = get_weeks_between(
      data[week_id_column],
      start_week_id - pretest_weeks,
      start_week_id - 1,
  )
  is_test = get_weeks_between(
      data[week_id_column],
      start_week_id,
      start_week_id + runtime_weeks - 1,
  )
  data[time_period_column] = None
  data.loc[is_pretest, time_period_column] = "pretest"
  data.loc[is_test, time_period_column] = "test"

  index_columns = [item_id_column]
  if treatment_column:
    index_columns.append(treatment_column)

  pivoted_data = pd.pivot_table(
      data,
      index=index_columns,
      columns=time_period_column,
      values=metric_columns,
      aggfunc="sum",
      fill_value=0.0,
  )

  if "pretest" in pivoted_data.columns.get_level_values(1):
    pivoted_data.loc[:, pd.IndexSlice[:, "pretest"]] /= pretest_weeks

  pivoted_data.loc[:, pd.IndexSlice[:, "test"]] /= runtime_weeks

  data.drop(columns=[time_period_column], inplace=True)

  return pivoted_data


def _pivot_crossover_time_assignment(
    data: pd.DataFrame,
    *,
    start_week_id: int,
    pretest_weeks: int,
    runtime_weeks: int,
    crossover_washout_weeks: int,
    metric_columns: list[str],
    item_id_column: str,
    week_id_column: str,
    treatment_column: str | None = None,
    time_period_column: str = "time_period",
) -> pd.DataFrame:
  """Pivots the data from a crossover A/B test so that it's ready for analysis.

  This function pivots the input dataset so that each row is an independent
  randomisation unit, and the columns are combinations of (metric, time_period).
  If treatment_column is not specified then the index is just the item_id,
  while if it is specified then the index is pairs of (item_id, treatment).

  The time_period is either pretest (for the period before the test began),
  test_1 (for the first period during the experiment, before the crossover) or
  test_2 (for the second period during the experiment, after the crossover).
  If there is no pretest period, then it will only contain the test_1 and
  test_2 periods.

  The values are the average of the metric per week during the time_period.

  Args:
    data: The data from the experiment to be pivoted.
    start_week_id: The id of the first week of the experiment (inclusive).
    pretest_weeks: The number of weeks to use as pre-test weeks.
    runtime_weeks: The number of weeks to use as experiment runtime.
    crossover_washout_weeks: The number of weeks to ignore from the analysis
      after the start of the experiment and after the crossover.
    metric_columns: The list of metrics to be analyzed.
    item_id_column: The column identifying individual randomisation units.
    week_id_column: The column identifying individual weeks.
    treatment_column: The column that indicates if the item_id is in the
      treatment or control group. None if there is no treatment column.
    time_period_column: The column name used internally to do the time
      assignment. It is not returned, and it should not already exist in the
      data.

  Returns:
    The pivoted data, where rows are randomisation units, and columns are
    combinations of metrics and time_periods.
  """
  test_period_weeks = (
      runtime_weeks - 2 * crossover_washout_weeks
  ) // 2
  pretest_start_week = start_week_id - pretest_weeks
  post_washout_period_1_start = (
      start_week_id + crossover_washout_weeks
  )
  period_1_end = post_washout_period_1_start + test_period_weeks - 1
  post_washout_period_2_start = (
      period_1_end + 1 + crossover_washout_weeks
  )
  period_2_end = post_washout_period_2_start + test_period_weeks - 1

  is_pretest = get_weeks_between(
      data[week_id_column], pretest_start_week, start_week_id - 1
  )
  is_test_1 = get_weeks_between(
      data[week_id_column],
      post_washout_period_1_start,
      period_1_end,
  )
  is_test_2 = get_weeks_between(
      data[week_id_column],
      post_washout_period_2_start,
      period_2_end,
  )
  data[time_period_column] = None
  data.loc[is_pretest, time_period_column] = "pretest"
  data.loc[is_test_1, time_period_column] = "test_1"
  data.loc[is_test_2, time_period_column] = "test_2"

  index_columns = [item_id_column]
  if treatment_column:
    index_columns.append(treatment_column)

  pivoted_data = pd.pivot_table(
      data,
      index=index_columns,
      columns=time_period_column,
      values=metric_columns,
      aggfunc="sum",
      fill_value=0,
  )

  if "pretest" in pivoted_data.columns.get_level_values(1):
    pivoted_data.loc[:, pd.IndexSlice[:, "pretest"]] /= pretest_weeks

  pivoted_data.loc[:, pd.IndexSlice[:, "test_1"]] /= test_period_weeks
  pivoted_data.loc[:, pd.IndexSlice[:, "test_2"]] /= test_period_weeks

  data.drop(columns=[time_period_column], inplace=True)

  return pivoted_data


def pivot_time_assignment(
    data: pd.DataFrame,
    *,
    design: ExperimentDesign,
    start_week_id: int,
    metric_columns: list[str],
    item_id_column: str,
    week_id_column: str,
    treatment_column: str | None = None,
    time_period_column: str = "time_period",
) -> pd.DataFrame:
  """Pivots the data so that it's ready for analysis.

  This function pivots the input dataset so that each row is an independent
  randomisation unit, and the columns are combinations of (metric, time_period).
  If treatment_column is not specified then the index is just the item_id,
  while if it is specified then the index is pairs of (item_id, treatment).

  If the experiment is a crossover design then the time_period is either pretest
  (for the period before the test began), test_1 (for the first period during
  the experiment, before the crossover) or test_2 (for the second period during
  the experiment, after the crossover). If there is no pretest period, then it
  will only contain the test_1 and test_2 periods.

  If the experiment is a regular (non-crossover) design then the time_period is
  either pretest (for the period before the test began), or test (for the period
  during the experiment). If there is no pretest period, then it will only
  contain the test period.

  The values are the average of the metric per week during the time_period.

  Args:
    data: The data from the experiment to be pivoted.
    design: The design of the experiment, containing the number of weeks to use
      for the pretest, runtime and washout periods and whether it's a crossover
      design.
    start_week_id: The id of the first week of the experiment (inclusive).
    metric_columns: The list of metrics to be analyzed.
    item_id_column: The column identifying individual randomisation units.
    week_id_column: The column identifying individual weeks.
    treatment_column: The column that indicates if the item_id is in the
      treatment or control group. None if there is no treatment column.
    time_period_column: The column name used internally to do the time
      assignment. It will be the name of the time_period index in the returned
      dataframe.

  Returns:
    The pivoted data, where rows are randomisation units, and columns are
    combinations of metrics and time_periods.

  Raises:
    ValueError: If the time_period_column exists in the data, or if the
      dataframe does not have the expected columns.
  """
  if time_period_column in data.columns:
    raise ValueError(
        "time_period_column must not already be a column in the data."
    )

  required_columns = set([item_id_column, week_id_column] + metric_columns)
  if treatment_column:
    required_columns.add(treatment_column)
  missing_columns = required_columns - set(data.columns)
  if missing_columns:
    raise ValueError(
        f"The dataframe does not have the following columns: {missing_columns}"
    )

  if design.is_crossover:
    return _pivot_crossover_time_assignment(
        data,
        start_week_id=start_week_id,
        pretest_weeks=design.pretest_weeks,
        runtime_weeks=design.runtime_weeks,
        crossover_washout_weeks=design.crossover_washout_weeks,
        metric_columns=metric_columns,
        item_id_column=item_id_column,
        week_id_column=week_id_column,
        treatment_column=treatment_column,
        time_period_column=time_period_column,
    )
  else:
    return _pivot_regular_time_assignment(
        data,
        start_week_id=start_week_id,
        pretest_weeks=design.pretest_weeks,
        runtime_weeks=design.runtime_weeks,
        metric_columns=metric_columns,
        item_id_column=item_id_column,
        week_id_column=week_id_column,
        treatment_column=treatment_column,
        time_period_column=time_period_column,
    )


def _analyze_regular_experiment(
    data: pd.DataFrame, post_trim_percentile: float
) -> statistics.StatisticalTestResults:
  """Analyzes a regular (non-crossover) A/B test.

  This applies Yuen's t-test to the dataset provided. If pretest data exists,
  then it also applies the cuped adjustment.

  Args:
    data: The dataset to be analyzed. Must contain the columns "test" and
      "treatment", and optionally "pretest", and each row must correspond to a
      different randomisation unit. The "test" column contains the value of the
      metric for that sample, and the "treatment" column is either 1 or 0 and
      flags whether the sample is in the control group (0) or treatment group
      (1). If it exists, the "pretest" column should contain the value of the
      metric for that sample from the period before the experiment began.
    post_trim_percentile: The fraction of smallest and largest samples to trim
      from the metric in the analysis.

  Returns:
    The statistical test results.
  """
  if "pretest" in data.columns:
    y = statistics.apply_cuped_adjustment(
        data["test"].values, data["pretest"].values, post_trim_percentile
    )
  else:
    y = data["test"].values

  y_0 = y[data["treatment"].values == 0]
  y_1 = y[data["treatment"].values == 1]

  return statistics.yuens_t_test_ind(
      y_1,
      y_0,
      trimming_quantile=post_trim_percentile,
      equal_var=False,
      alternative="two-sided",
  )


def _analyze_crossover_experiment(
    data: pd.DataFrame, post_trim_percentile: float
) -> statistics.StatisticalTestResults:
  """Analyzes a crossover A/B test.

  This applies Yuen's t-test to the dataset provided.

  Args:
    data: The dataset to be analyzed. Must contain the columns "test_1",
      "test_2" and "treatment", and each row must correspond to a different
      randomisation unit. The "test_1" column contains the value of the metric
      for that sample from the first period of the crossover test, and the
      "test_2" column contains the value of the metric from the second period of
      the crossover test. The "treatment" column is either 1 or 0 and flags
      whether the sample was treated in the first period (1) or in the second
      period (0).
    post_trim_percentile: The fraction of smallest and largest samples to trim
      from the metric in the analysis.

  Returns:
    The statistical test results.
  """
  y_1 = data["test_1"].values
  y_2 = data["test_2"].values
  is_treated_first = data["treatment"].values == 1

  # I need to align the means of the first and second period to reduce variance,
  # but for the relative uplift to make sense I will align them to the
  # control mean, not to 0.0. Therefore I will first set the means to 0,
  # then calculate treatment and control, and then adjust the means again
  # so that the control mean matches the original control mean.
  y_1_demeaned = y_1 - y_1.mean()
  y_2_demeaned = y_2 - y_2.mean()

  # The control data is the "treated first" group in the second period,
  # and the "not treated first" group in the first period.
  y_control_demeaned = y_1_demeaned.copy()
  y_control_demeaned[is_treated_first] = y_2_demeaned[is_treated_first].copy()

  # The treatment data is the "treated first" group in the first period,
  # and the "not treated first" group in the second period.
  y_treatment_demeaned = y_1_demeaned.copy()
  y_treatment_demeaned[~is_treated_first] = y_2_demeaned[
      ~is_treated_first
  ].copy()

  # Now adjust the mean again so that the adjusted control mean is the same
  # as the original control mean
  y_control = y_1.copy()
  y_control[is_treated_first] = y_2[is_treated_first].copy()
  adjustment_factor = y_control.mean() - y_control_demeaned.mean()
  y_control_demeaned += adjustment_factor
  y_treatment_demeaned += adjustment_factor

  return statistics.yuens_t_test_paired(
      y_treatment_demeaned,
      y_control_demeaned,
      trimming_quantile=post_trim_percentile,
      alternative="two-sided",
  )


def analyze_experiment(
    data: pd.DataFrame, design: ExperimentDesign
) -> statistics.StatisticalTestResults:
  """Analyzes an A/B test.

  This applies Yuen's t-test to the dataset provided, where the dataset is
  typically generated with pivot_time_assignment().

  If you are analyzing a crossover experiment, then data must contain the
  columns "test_1", "test_2" and "treatment", and each row must correspond to a
  different randomisation unit. The "test_1" column contains the value of the
  metric for that sample from the first period of the crossover test, and the
  "test_2" column contains the value of the metric from the second period of the
  crossover test. The "treatment" column is either 1 or 0 and flags whether the
  sample was treated in the first period (1) or in the second period (0).

  If you are analyzing a regular (non-crossover) experiment, then the data must
  contain the columns "test" and "treatment", and optionally "pretest", and each
  row must correspond to a different randomisation unit. The "test" column
  contains the value of the metric for that sample, and the "treatment" column
  is either 1 or 0 and flags whether the sample is in the control group (0) or
  treatment group (1). If it exists, the "pretest" column should contain the
  value of the metric for that sample from the period before the experiment
  began.

  Args:
    data: The dataset to be analyzed.
    design: The design of the experiment being analyzed, containing the post
      trim percentile and whether the it is a crossover test or not.

  Returns:
    The statistical test results.

  Raises:
    ValueError: If the dataframe does not have the required columns (depending
      on the design).
  """
  required_columns = {"treatment"}
  if design.is_crossover:
    required_columns.update({"test_1", "test_2"})
  else:
    required_columns.add("test")
    if design.pretest_weeks > 0:
      required_columns.add("pretest")

  missing_columns = required_columns - set(data.columns)
  if missing_columns:
    raise ValueError(
        "The dataframe is missing the following required columns: "
        f"{missing_columns}"
    )

  if design.is_crossover:
    return _analyze_crossover_experiment(data, design.post_trim_percentile)
  else:
    return _analyze_regular_experiment(data, design.post_trim_percentile)
