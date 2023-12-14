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

import numpy as np
import pandas as pd

from feedx import experiment_design
from feedx import statistics

ExperimentDesign = experiment_design.ExperimentDesign


def get_weeks_between(
    week_id: pd.Series, start_id: int, end_id: int
) -> pd.Series:
  """Returns a boolean array, true if week_id is within the range.

  Returns true if the week_id is between start_id and end_id (both inclusive).

  This function will be used to assign treatment time to items based on the
  selected number of runtime weeks and regular vs crossover experiment.

  Args:
    week_id: week id associated with a specific date in the dataset.
    start_id: integer value representing the starting point of a test phase.
    end_id: integer value representing the ending point of a test phase.
  """
  greater = week_id >= start_id
  less = week_id <= end_id
  return greater & less


def _add_time_period_column_for_regular_experiment(
    data: pd.DataFrame,
    *,
    start_week_id: int,
    pretest_weeks: int,
    runtime_weeks: int,
    week_id_column: str,
    time_period_column: str = "time_period",
) -> pd.DataFrame:
  """Adds the time_period_column to the data for a regular experiment."""
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
  return data


def _add_time_period_column_for_crossover_experiment(
    data: pd.DataFrame,
    *,
    start_week_id: int,
    pretest_weeks: int,
    runtime_weeks: int,
    crossover_washout_weeks: int,
    week_id_column: str,
    time_period_column: str = "time_period",
) -> pd.DataFrame:
  """Adds the time_period_column to the data for a crossover experiment."""
  test_period_weeks = (runtime_weeks - 2 * crossover_washout_weeks) // 2
  pretest_start_week = start_week_id - pretest_weeks
  post_washout_period_1_start = start_week_id + crossover_washout_weeks
  period_1_end = post_washout_period_1_start + test_period_weeks - 1
  post_washout_period_2_start = period_1_end + 1 + crossover_washout_weeks
  period_2_end = post_washout_period_2_start + test_period_weeks - 1

  is_pretest = get_weeks_between(
      data[week_id_column], pretest_start_week, start_week_id - 1
  )
  is_washout_1 = get_weeks_between(
      data[week_id_column],
      start_week_id,
      post_washout_period_1_start - 1,
  )
  is_test_1 = get_weeks_between(
      data[week_id_column],
      post_washout_period_1_start,
      period_1_end,
  )
  is_washout_2 = get_weeks_between(
      data[week_id_column],
      period_1_end + 1,
      post_washout_period_2_start - 1,
  )
  is_test_2 = get_weeks_between(
      data[week_id_column],
      post_washout_period_2_start,
      period_2_end,
  )
  data[time_period_column] = None
  data.loc[is_pretest, time_period_column] = "pretest"
  data.loc[is_washout_1, time_period_column] = "washout_1"
  data.loc[is_test_1, time_period_column] = "test_1"
  data.loc[is_washout_2, time_period_column] = "washout_2"
  data.loc[is_test_2, time_period_column] = "test_2"

  return data


def add_time_period_column(
    data: pd.DataFrame,
    *,
    design: ExperimentDesign,
    start_week_id: int,
    week_id_column: str,
    time_period_column: str = "time_period",
):
  """Adds the time_period_column to the data.

  If the experiment is a crossover design then the time_period is either pretest
  (for the period before the test began), washout_1 (for the first period of the
  runtime to exclude from the analysis), test_1 (for the first period during
  the experiment, before the crossover), washout_2 (for the first period after
  the crossover to exclude from the analysis) or test_2 (for the second period
  during the experiment, after the crossover). It is None if it's none of those
  periods.

  If the experiment is a regular (non-crossover) design then the time_period is
  either pretest (for the period before the test began), or test (for the period
  during the experiment). It is None if it's none of those periods.

  The values are the average of the metric per week during the time_period.

  Args:
    data: The data from the experiment to be pivoted.
    design: The design of the experiment, containing the number of weeks to use
      for the pretest, runtime and washout periods and whether it's a crossover
      design.
    start_week_id: The id of the first week of the experiment (inclusive).
    week_id_column: The column identifying individual weeks.
    time_period_column: The output column name containing the assigned time
      period. Defaults to "time_period".

  Returns:
    The data with the time period column added.

  Raises:
    ValueError: If the time_period_column exists in the data, or if the
      week_id_column is missing from the data.
  """
  if time_period_column in data.columns:
    raise ValueError(
        "time_period_column must not already be a column in the data."
    )
  if week_id_column not in data.columns:
    raise ValueError("week_id_column must be a column in the data.")

  if design.is_crossover:
    return _add_time_period_column_for_crossover_experiment(
        data,
        start_week_id=start_week_id,
        pretest_weeks=design.pretest_weeks,
        runtime_weeks=design.runtime_weeks,
        crossover_washout_weeks=design.crossover_washout_weeks,
        week_id_column=week_id_column,
        time_period_column=time_period_column,
    )
  else:
    return _add_time_period_column_for_regular_experiment(
        data,
        start_week_id=start_week_id,
        pretest_weeks=design.pretest_weeks,
        runtime_weeks=design.runtime_weeks,
        week_id_column=week_id_column,
        time_period_column=time_period_column,
    )


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
  data = _add_time_period_column_for_regular_experiment(
      data,
      start_week_id=start_week_id,
      pretest_weeks=pretest_weeks,
      runtime_weeks=runtime_weeks,
      week_id_column=week_id_column,
      time_period_column=time_period_column,
  )

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
  data = _add_time_period_column_for_crossover_experiment(
      data,
      start_week_id=start_week_id,
      pretest_weeks=pretest_weeks,
      runtime_weeks=runtime_weeks,
      crossover_washout_weeks=crossover_washout_weeks,
      week_id_column=week_id_column,
      time_period_column=time_period_column,
  )
  test_period_weeks = (runtime_weeks - 2 * crossover_washout_weeks) // 2

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

  if "test_1" in pivoted_data.columns.get_level_values(1):
    pivoted_data.loc[:, pd.IndexSlice[:, "test_1"]] /= test_period_weeks
  if "test_2" in pivoted_data.columns.get_level_values(1):
    pivoted_data.loc[:, pd.IndexSlice[:, "test_2"]] /= test_period_weeks
  if "washout_1" in pivoted_data.columns.get_level_values(1):
    pivoted_data.loc[
        :, pd.IndexSlice[:, "washout_1"]
    ] /= crossover_washout_weeks
  if "washout_2" in pivoted_data.columns.get_level_values(1):
    pivoted_data.loc[
        :, pd.IndexSlice[:, "washout_2"]
    ] /= crossover_washout_weeks

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


def _prepare_metrics_for_regular_experiment(
    data: pd.DataFrame,
    treatment_assignment_column: str,
    post_trim_percentile: float,
) -> tuple[np.ndarray, np.ndarray]:
  """Splits the metric into control and treatment and applies cuped."""
  if "pretest" in data.columns:
    y = statistics.apply_cuped_adjustment(
        data["test"].values, data["pretest"].values, post_trim_percentile
    )
  else:
    y = data["test"].values

  y_control = y[data[treatment_assignment_column].values == 0]
  y_treatment = y[data[treatment_assignment_column].values == 1]
  return y_treatment, y_control


def _analyze_regular_experiment_single_metric(
    data: pd.DataFrame,
    *,
    denominator_data: pd.DataFrame | None,
    post_trim_percentile: float,
    treatment_assignment_column: str,
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
    denominator_data: Same as data, but for the denominator metric if analysing
      a ratio.
    post_trim_percentile: The fraction of smallest and largest samples to trim
      from the metric in the analysis.
    treatment_assignment_column: The column containin the treatment assignment
      in the data.

  Returns:
    The statistical test results.
  """
  y_treatment, y_control = _prepare_metrics_for_regular_experiment(
      data, treatment_assignment_column, post_trim_percentile
  )

  if denominator_data is not None:
    denominator_y_treatment, denominator_y_control = (
        _prepare_metrics_for_regular_experiment(
            denominator_data, treatment_assignment_column, post_trim_percentile
        )
    )
  else:
    denominator_y_control = None
    denominator_y_treatment = None

  return statistics.yuens_t_test_ind(
      y_treatment,
      y_control,
      trimming_quantile=post_trim_percentile,
      denom_values1=denominator_y_treatment,
      denom_values2=denominator_y_control,
      equal_var=False,
      alternative="two-sided",
  )


def _prepare_metrics_for_crossover_experiment(
    data: pd.DataFrame,
    treatment_assignment_column: str,
) -> tuple[np.ndarray, np.ndarray]:
  """Demeans the data and aligns the treatment and control time periods."""
  y_1 = data["test_1"].values
  y_2 = data["test_2"].values
  is_treated_first = data[treatment_assignment_column].values == 1

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

  return y_treatment_demeaned, y_control_demeaned


def _analyze_crossover_experiment_single_metric(
    data: pd.DataFrame,
    *,
    denominator_data: pd.DataFrame | None,
    post_trim_percentile: float,
    treatment_assignment_column: str,
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
    denominator_data: Same as data, but for the denominator metric if analysing
      a ratio.
    post_trim_percentile: The fraction of smallest and largest samples to trim
      from the metric in the analysis.
    treatment_assignment_column: The column containin the treatment assignment
      in the data.

  Returns:
    The statistical test results.
  """

  y_treatment_demeaned, y_control_demeaned = (
      _prepare_metrics_for_crossover_experiment(
          data, treatment_assignment_column
      )
  )

  if denominator_data is not None:
    denominator_y_treatment_demeaned, denominator_y_control_demeaned = (
        _prepare_metrics_for_crossover_experiment(
            denominator_data, treatment_assignment_column
        )
    )
  else:
    denominator_y_treatment_demeaned = None
    denominator_y_control_demeaned = None

  return statistics.yuens_t_test_paired(
      y_treatment_demeaned,
      y_control_demeaned,
      trimming_quantile=post_trim_percentile,
      denom_values1=denominator_y_treatment_demeaned,
      denom_values2=denominator_y_control_demeaned,
      alternative="two-sided",
  )


def analyze_single_metric(
    data: pd.DataFrame,
    *,
    design: ExperimentDesign,
    metric_name: str,
    denominator_metric_name: str | None = None,
    treatment_assignment_index_name: str = "treatment_assignment",
    apply_trimming_if_in_design: bool = True,
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

  If you specify a denominator_metric_name, then the analysis will be done for a
  ratio metric, where metric_name is the numerator and denominator_metric_name
  is the denominator.

  Args:
    data: The dataset to be analyzed. This should have been generated with
      pivot_time_assignment(), so it will have a two level index containing the
      item_id and the treatment assignment, and a two level column where the top
      level is the name of the metric, and the second level is the time period.
    design: The design of the experiment being analyzed, containing the post
      trim percentile and whether the it is a crossover test or not.
    metric_name: The name of the metric to be analyzed. This should exist as the
      top level
    denominator_metric_name: The name of the metric that is the denominator if a
      ratio is to be analyzed. Defaults to None, meaning the analysis is for a
      regular, non-ratio metric.
    treatment_assignment_index_name: The name of the index level containing the
      treatment assignment in the data. Defaults to "treatment_assignment".
    apply_trimming_if_in_design: Whether to apply the trimming if it is
      specified in the design. Defaults to True.

  Returns:
    The statistical test results.

  Raises:
    ValueError: If the dataframe does not have the required columns (depending
      on the design).
  """
  data_metric_subset = data[metric_name].reset_index()
  if denominator_metric_name is not None:
    denominator_metric_subset = data[denominator_metric_name].reset_index()
  else:
    denominator_metric_subset = None

  required_columns = {treatment_assignment_index_name}
  if design.is_crossover:
    required_columns.update({"test_1", "test_2"})
  else:
    required_columns.add("test")
    if design.pretest_weeks > 0:
      required_columns.add("pretest")

  missing_columns = required_columns - set(data_metric_subset.columns)
  if denominator_metric_name is not None:
    missing_columns.update(
        required_columns - set(denominator_metric_subset.columns)
    )

  if missing_columns:
    raise ValueError(
        "The dataframe is missing the following required columns: "
        f"{missing_columns}"
    )

  if apply_trimming_if_in_design:
    post_trim_percentile = design.post_trim_percentile
  else:
    post_trim_percentile = 0.0

  if design.is_crossover:
    return _analyze_crossover_experiment_single_metric(
        data_metric_subset,
        denominator_data=denominator_metric_subset,
        post_trim_percentile=post_trim_percentile,
        treatment_assignment_column=treatment_assignment_index_name,
    )
  else:
    return _analyze_regular_experiment_single_metric(
        data_metric_subset,
        denominator_data=denominator_metric_subset,
        post_trim_percentile=post_trim_percentile,
        treatment_assignment_column=treatment_assignment_index_name,
    )
