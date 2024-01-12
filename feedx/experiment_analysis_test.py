# Copyright 2024 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
from typing import List
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pandas as pd

from feedx import experiment_analysis
from feedx import experiment_design
from feedx import statistics

StatisticalTestResults = experiment_analysis.statistics.StatisticalTestResults


class GetWeeksBetweenTest(parameterized.TestCase):

  def test_get_weeks_between(self):
    values = [
        ["ABC123", "2023-10-02", 1, 2314, 324523],
        ["XYZ987", "2023-10-01", 4, 1, 10000],
        ["DEF123", "2023-10-02", 5, 2314, 324523],
        ["UVW987", "2023-10-01", 8, 1, 10000],
        ["GHI123", "2023-10-01", 10, 1, 10000],
    ]

    data_frame = pd.DataFrame(
        values,
        columns=["item_id", "date", "week_number", "clicks", "impressions"],
    )

    actual_result = experiment_analysis.get_weeks_between(
        data_frame["week_number"], start_id=4, end_id=8
    )
    expected_result = [False, True, True, True, False]

    self.assertListEqual(actual_result.values.tolist(), expected_result)


class AnalyzeSingleMetricTests(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.irrelevant_design_args = dict(
        n_items_before_trimming=1000,
        runtime_weeks=4,
        primary_metric="clicks",
        pre_trim_top_percentile=0.0,
        pre_trim_bottom_percentile=0.0,
        crossover_washout_weeks=1,
    )

  def test_analyze_regular_experiment_returns_analysis_results(self):
    data = pd.DataFrame({
        "treatment_assignment": [0] * 5 + [1] * 5,
        "test": [0.2, 0.5, 0.6, 0.3, 0.1, 0.15, 0.64, 0.3, 0.2, 0.9],
        "pretest": [0.4, 0.2, 0.6, 0.1, 0.2, 0.4, 0.2, 0.6, 0.1, 0.2],
    }).set_index(["treatment_assignment"])
    data.columns = pd.MultiIndex.from_product([["clicks"], data.columns])

    design = experiment_design.ExperimentDesign(
        is_crossover=False,
        post_trim_percentile=0.1,
        pretest_weeks=4,
        **self.irrelevant_design_args
    )

    analysis_result = experiment_analysis.analyze_single_metric(
        data,
        design=design,
        metric_name="clicks",
    )

    self.assertIsInstance(analysis_result, StatisticalTestResults)

  def test_analyze_regular_experiment_ratio_metric_returns_analysis_results(
      self,
  ):
    data = pd.DataFrame({
        "treatment_assignment": [0] * 5 + [1] * 5,
        "test": [0.2, 0.5, 0.6, 0.3, 0.1, 0.15, 0.64, 0.3, 0.2, 0.9],
        "pretest": [0.4, 0.2, 0.6, 0.1, 0.2, 0.4, 0.2, 0.6, 0.1, 0.2],
    }).set_index(["treatment_assignment"])
    data.columns = pd.MultiIndex.from_product([["clicks"], data.columns])

    denominator_data = pd.DataFrame({
        "treatment_assignment": [0] * 5 + [1] * 5,
        "test": [0.4, 0.3, 0.7, 1.1, 1.5, 0.8, 0.3, 0.2, 0.2, 0.5],
        "pretest": [0.3, 0.6, 0.7, 0.2, 2.0, 7.0, 2.0, 1.0, 0.4, 0.1],
    }).set_index(["treatment_assignment"])
    denominator_data.columns = pd.MultiIndex.from_product(
        [["impressions"], denominator_data.columns]
    )

    data = pd.concat([data, denominator_data], axis=1)

    design = experiment_design.ExperimentDesign(
        is_crossover=False,
        post_trim_percentile=0.1,
        pretest_weeks=4,
        **self.irrelevant_design_args
    )

    analysis_result = experiment_analysis.analyze_single_metric(
        data,
        design=design,
        metric_name="clicks",
        denominator_metric_name="impressions",
    )

    self.assertIsInstance(analysis_result, StatisticalTestResults)

  def test_analyze_regular_experiment_uses_cuped_when_pretest_is_provided(self):
    data = pd.DataFrame({
        "treatment_assignment": [0] * 5 + [1] * 5,
        "test": [0.2, 0.5, 0.6, 0.3, 0.1, 0.15, 0.64, 0.3, 0.2, 0.9],
        "pretest": [0.4, 0.2, 0.6, 0.1, 0.2, 0.4, 0.2, 0.6, 0.1, 0.2],
    }).set_index(["treatment_assignment"])
    data.columns = pd.MultiIndex.from_product([["clicks"], data.columns])

    design = experiment_design.ExperimentDesign(
        is_crossover=False,
        post_trim_percentile=0.1,
        pretest_weeks=4,
        **self.irrelevant_design_args
    )

    with mock.patch(
        "feedx.statistics.apply_cuped_adjustment",
        side_effect=lambda x, *y: x,
    ) as mock_apply_cuped_adjustment:
      experiment_analysis.analyze_single_metric(
          data,
          design=design,
          metric_name="clicks",
      )
      mock_apply_cuped_adjustment.assert_called_once()

  def test_analyze_regular_experiment_does_not_use_cuped_when_pretest_is_not_provided(
      self,
  ):
    data = pd.DataFrame({
        "treatment_assignment": [0] * 5 + [1] * 5,
        "test": [0.2, 0.5, 0.6, 0.3, 0.1, 0.15, 0.64, 0.3, 0.2, 0.9],
    }).set_index(["treatment_assignment"])
    data.columns = pd.MultiIndex.from_product([["clicks"], data.columns])

    design = experiment_design.ExperimentDesign(
        is_crossover=False,
        post_trim_percentile=0.1,
        pretest_weeks=0,
        **self.irrelevant_design_args
    )

    with mock.patch(
        "feedx.statistics.apply_cuped_adjustment"
    ) as mock_apply_cuped_adjustment:
      experiment_analysis.analyze_single_metric(
          data,
          design=design,
          metric_name="clicks",
      )
      mock_apply_cuped_adjustment.assert_not_called()

  def test_analyze_crossover_experiment_returns_analysis_results(self):
    data = pd.DataFrame({
        "treatment_assignment": [0] * 5 + [1] * 5,
        "test_1": [0.2, 0.5, 0.6, 0.3, 0.1, 0.15, 0.64, 0.3, 0.2, 0.9],
        "test_2": [0.4, 0.2, 0.6, 0.1, 0.2, 0.4, 0.2, 0.6, 0.1, 0.2],
    }).set_index(["treatment_assignment"])
    data.columns = pd.MultiIndex.from_product([["clicks"], data.columns])

    design = experiment_design.ExperimentDesign(
        is_crossover=True,
        post_trim_percentile=0.1,
        pretest_weeks=4,
        **self.irrelevant_design_args
    )

    analysis_result = experiment_analysis.analyze_single_metric(
        data,
        design=design,
        metric_name="clicks",
    )

    self.assertIsInstance(analysis_result, StatisticalTestResults)

  def test_analyze_crossover_experiment_ratio_metric_returns_analysis_results(
      self,
  ):
    data = pd.DataFrame({
        "treatment_assignment": [0] * 5 + [1] * 5,
        "test_1": [0.2, 0.5, 0.6, 0.3, 0.1, 0.15, 0.64, 0.3, 0.2, 0.9],
        "test_2": [0.4, 0.2, 0.6, 0.1, 0.2, 0.4, 0.2, 0.6, 0.1, 0.2],
    }).set_index(["treatment_assignment"])
    data.columns = pd.MultiIndex.from_product([["clicks"], data.columns])

    denominator_data = pd.DataFrame({
        "treatment_assignment": [0] * 5 + [1] * 5,
        "test_1": [0.4, 0.3, 0.7, 1.1, 1.5, 0.8, 0.3, 0.2, 0.2, 0.5],
        "test_2": [0.3, 0.6, 0.7, 0.2, 2.0, 7.0, 2.0, 1.0, 0.4, 0.1],
    }).set_index(["treatment_assignment"])
    denominator_data.columns = pd.MultiIndex.from_product(
        [["impressions"], denominator_data.columns]
    )

    data = pd.concat([data, denominator_data], axis=1)

    design = experiment_design.ExperimentDesign(
        is_crossover=True,
        post_trim_percentile=0.1,
        pretest_weeks=4,
        **self.irrelevant_design_args
    )

    analysis_result = experiment_analysis.analyze_single_metric(
        data,
        design=design,
        metric_name="clicks",
        denominator_metric_name="impressions",
    )

    self.assertIsInstance(analysis_result, StatisticalTestResults)

  def test_analyze_crossover_experiment_returns_control_average_when_no_trimming_is_applied(
      self
  ):
    data = pd.DataFrame({
        "treatment_assignment": [0] * 5 + [1] * 5,
        "test_1": [1.0, 2.0, 10.0, 1.0, 5.0, 20.0, 15.0, 11.0, 1.0, 1.0],
        "test_2": [5.0, 7.0, 14.0, 3.0, 6.0, 40.0, 16.0, 9.0, 3.0, 2.0],
    }).set_index(["treatment_assignment"])
    data.columns = pd.MultiIndex.from_product([["clicks"], data.columns])

    design = experiment_design.ExperimentDesign(
        is_crossover=True,
        post_trim_percentile=0.0,
        pretest_weeks=4,
        **self.irrelevant_design_args
    )

    analysis_result = experiment_analysis.analyze_single_metric(
        data,
        design=design,
        metric_name="clicks",
    )

    data_subset = data["clicks"].reset_index()
    expected_control_average = (
        data_subset.loc[
            data_subset["treatment_assignment"] == 0, "test_1"
        ].sum()
        + data_subset.loc[
            data_subset["treatment_assignment"] == 1, "test_2"
        ].sum()
    ) / len(data_subset.index.values)

    self.assertAlmostEqual(
        analysis_result.control_average, expected_control_average
    )

  def test_analyze_regular_experiment_returns_control_average_when_no_trimming_is_applied(
      self,
  ):
    data = pd.DataFrame({
        "treatment_assignment": [0] * 5 + [1] * 5,
        "test": [0.2, 0.5, 0.6, 0.3, 0.1, 0.15, 0.64, 0.3, 0.2, 0.9],
    }).set_index(["treatment_assignment"])
    data.columns = pd.MultiIndex.from_product([["clicks"], data.columns])

    design = experiment_design.ExperimentDesign(
        is_crossover=False,
        post_trim_percentile=0.0,
        pretest_weeks=0,
        **self.irrelevant_design_args
    )

    analysis_result = experiment_analysis.analyze_single_metric(
        data,
        design=design,
        metric_name="clicks",
    )

    data_subset = data["clicks"].reset_index()
    expected_control_average = data_subset.loc[
        data_subset["treatment_assignment"] == 0, "test"
    ].mean()

    self.assertEqual(analysis_result.control_average, expected_control_average)

  @parameterized.parameters(True, False)
  def test_raises_exception_if_treatment_assignment_column_is_missing(
      self, is_crossover
  ):
    data = pd.DataFrame({
        "treatment": [0] * 5 + [1] * 5,
        "test_1": [0.2, 0.5, 0.6, 0.3, 0.1, 0.15, 0.64, 0.3, 0.2, 0.9],
        "test_2": [0.4, 0.2, 0.6, 0.1, 0.2, 0.4, 0.2, 0.6, 0.1, 0.2],
    }).set_index(["treatment"])

    data.columns = pd.MultiIndex.from_product([["clicks"], data.columns])

    design = experiment_design.ExperimentDesign(
        is_crossover=is_crossover,
        post_trim_percentile=0.1,
        pretest_weeks=4,
        **self.irrelevant_design_args
    )

    with self.assertRaises(ValueError):
      experiment_analysis.analyze_single_metric(
          data,
          design=design,
          metric_name="clicks",
      )

  @parameterized.parameters("test_1", "test_2")
  def test_raises_exception_if_required_columns_for_crossover_are_missing(
      self, required_missing_column
  ):
    data = pd.DataFrame({
        "treatment_assignment": [0] * 5 + [1] * 5,
        "test_1": [0.2, 0.5, 0.6, 0.3, 0.1, 0.15, 0.64, 0.3, 0.2, 0.9],
        "test_2": [0.4, 0.2, 0.6, 0.1, 0.2, 0.4, 0.2, 0.6, 0.1, 0.2],
    }).set_index(["treatment_assignment"])

    data.drop(columns=required_missing_column, inplace=True)
    data.columns = pd.MultiIndex.from_product([["clicks"], data.columns])

    design = experiment_design.ExperimentDesign(
        is_crossover=True,
        post_trim_percentile=0.1,
        pretest_weeks=4,
        **self.irrelevant_design_args
    )

    with self.assertRaises(ValueError):
      experiment_analysis.analyze_single_metric(
          data,
          design=design,
          metric_name="clicks",
      )

  @parameterized.parameters("test", "pretest")
  def test_raises_exception_if_required_columns_for_regular_experiment_are_missing(
      self, required_missing_column
  ):
    data = pd.DataFrame({
        "treatment_assignment": [0] * 5 + [1] * 5,
        "test": [0.2, 0.5, 0.6, 0.3, 0.1, 0.15, 0.64, 0.3, 0.2, 0.9],
        "pretest": [0.4, 0.2, 0.6, 0.1, 0.2, 0.4, 0.2, 0.6, 0.1, 0.2],
    }).set_index(["treatment_assignment"])

    data.drop(columns=required_missing_column, inplace=True)
    data.columns = pd.MultiIndex.from_product([["clicks"], data.columns])

    design = experiment_design.ExperimentDesign(
        is_crossover=False,
        post_trim_percentile=0.1,
        pretest_weeks=4,
        **self.irrelevant_design_args
    )

    with self.assertRaises(ValueError):
      experiment_analysis.analyze_single_metric(
          data,
          design=design,
          metric_name="clicks",
      )

  def test_pretest_column_not_required_if_pretest_weeks_is_0(self):
    data = pd.DataFrame({
        "treatment_assignment": [0] * 5 + [1] * 5,
        "test": [0.2, 0.5, 0.6, 0.3, 0.1, 0.15, 0.64, 0.3, 0.2, 0.9],
    }).set_index(["treatment_assignment"])
    data.columns = pd.MultiIndex.from_product([["clicks"], data.columns])

    design = experiment_design.ExperimentDesign(
        is_crossover=False,
        post_trim_percentile=0.1,
        pretest_weeks=0,
        **self.irrelevant_design_args
    )

    analysis_results = experiment_analysis.analyze_single_metric(
        data,
        design=design,
        metric_name="clicks",
    )

    self.assertIsInstance(analysis_results, StatisticalTestResults)

  @parameterized.parameters(False, True)
  def test_analyze_single_metric_does_not_mutate_original_dataframe(
      self, is_crossover
  ):
    data = pd.DataFrame({
        "treatment_assignment": [0] * 5 + [1] * 5,
        "pretest": [0.4, 0.2, 0.6, 0.1, 0.2, 0.4, 0.2, 0.6, 0.1, 0.2],
        "test": [0.2, 0.5, 0.6, 0.3, 0.1, 0.15, 0.64, 0.3, 0.2, 0.9],
        "test_1": [0.2, 0.5, 0.6, 0.3, 0.1, 0.15, 0.64, 0.3, 0.2, 0.9],
        "test_2": [0.4, 0.2, 0.6, 0.1, 0.2, 0.4, 0.2, 0.6, 0.1, 0.2],
    }).set_index(["treatment_assignment"])
    data.columns = pd.MultiIndex.from_product([["clicks"], data.columns])

    original_data_copy = data.copy()

    design = experiment_design.ExperimentDesign(
        is_crossover=is_crossover,
        post_trim_percentile=0.1,
        pretest_weeks=4,
        **self.irrelevant_design_args
    )

    experiment_analysis.analyze_single_metric(
        data,
        design=design,
        metric_name="clicks",
    )

    pd.testing.assert_frame_equal(data, original_data_copy)

  @parameterized.parameters(False, True)
  def test_analyze_single_metric_overrides_trimming_from_design_if_trimming_quantile_override_is_not_none(
      self, is_crossover
  ):
    data = pd.DataFrame({
        "treatment_assignment": [0] * 5 + [1] * 5,
        "pretest": [0.4, 0.2, 0.6, 0.1, 0.2, 0.4, 0.2, 0.6, 0.1, 0.2],
        "test": [0.2, 0.5, 0.6, 0.3, 0.1, 0.15, 0.64, 0.3, 0.2, 0.9],
        "test_1": [0.2, 0.5, 0.6, 0.3, 0.1, 0.15, 0.64, 0.3, 0.2, 0.9],
        "test_2": [0.4, 0.2, 0.6, 0.1, 0.2, 0.4, 0.2, 0.6, 0.1, 0.2],
    }).set_index(["treatment_assignment"])
    data.columns = pd.MultiIndex.from_product([["clicks"], data.columns])

    design = experiment_design.ExperimentDesign(
        is_crossover=is_crossover,
        post_trim_percentile=0.1,
        pretest_weeks=4,
        **self.irrelevant_design_args
    )
    with mock.patch(
        "feedx.statistics.TrimmedArray",
        side_effect=statistics.TrimmedArray,
    ) as mock_trimmed_array:
      experiment_analysis.analyze_single_metric(
          data,
          design=design,
          metric_name="clicks",
          trimming_quantile_override=0.05,
      )
      trimming_quantiles_used = set(
          [call[0][1] for call in mock_trimmed_array.call_args_list]
      )
      self.assertSetEqual(trimming_quantiles_used, {0.05})


class AddTimePeriodColumnTests(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.data = pd.DataFrame({
        "week_id": [-2, -1, 0, 1, 2, 3, 4],
    })
    self.irrelevant_design_args = dict(
        n_items_before_trimming=1000,
        primary_metric="clicks",
        pre_trim_top_percentile=0.0,
        pre_trim_bottom_percentile=0.0,
        post_trim_percentile=0.1,
    )

  def test_expected_time_period_added_for_regular_experiment_without_pretest(
      self,
  ):
    design = experiment_design.ExperimentDesign(
        is_crossover=False,
        runtime_weeks=4,
        pretest_weeks=0,
        **self.irrelevant_design_args
    )

    actual_output = experiment_analysis.add_time_period_column(
        self.data,
        design=design,
        start_week_id=0,
        week_id_column="week_id",
    )

    expected_output = pd.DataFrame({
        "week_id": [-2, -1, 0, 1, 2, 3, 4],
        "time_period": [None, None, "test", "test", "test", "test", None],
    })
    pd.testing.assert_frame_equal(
        actual_output, expected_output, check_like=True
    )

  def test_expected_time_period_added_for_regular_experiment_with_pretest(self):
    design = experiment_design.ExperimentDesign(
        is_crossover=False,
        runtime_weeks=4,
        pretest_weeks=2,
        **self.irrelevant_design_args
    )

    actual_output = experiment_analysis.add_time_period_column(
        self.data,
        design=design,
        start_week_id=0,
        week_id_column="week_id",
    )

    expected_output = pd.DataFrame({
        "week_id": [-2, -1, 0, 1, 2, 3, 4],
        "time_period": [
            "pretest",
            "pretest",
            "test",
            "test",
            "test",
            "test",
            None,
        ],
    })
    pd.testing.assert_frame_equal(
        actual_output, expected_output, check_like=True
    )

  def test_expected_time_period_added_for_crossover_experiment_without_pretest(
      self,
  ):
    design = experiment_design.ExperimentDesign(
        is_crossover=True,
        runtime_weeks=4,
        pretest_weeks=0,
        crossover_washout_weeks=1,
        **self.irrelevant_design_args
    )

    actual_output = experiment_analysis.add_time_period_column(
        self.data,
        design=design,
        start_week_id=0,
        week_id_column="week_id",
    )

    expected_output = pd.DataFrame({
        "week_id": [-2, -1, 0, 1, 2, 3, 4],
        "time_period": [
            None,
            None,
            "washout_1",
            "test_1",
            "washout_2",
            "test_2",
            None,
        ],
    })
    pd.testing.assert_frame_equal(
        actual_output, expected_output, check_like=True
    )

  def test_expected_time_period_added_for_crossover_experiment_with_pretest(
      self,
  ):
    design = experiment_design.ExperimentDesign(
        is_crossover=True,
        runtime_weeks=4,
        pretest_weeks=2,
        crossover_washout_weeks=1,
        **self.irrelevant_design_args
    )

    actual_output = experiment_analysis.add_time_period_column(
        self.data,
        design=design,
        start_week_id=0,
        week_id_column="week_id",
    )

    expected_output = pd.DataFrame({
        "week_id": [-2, -1, 0, 1, 2, 3, 4],
        "time_period": [
            "pretest",
            "pretest",
            "washout_1",
            "test_1",
            "washout_2",
            "test_2",
            None,
        ],
    })
    pd.testing.assert_frame_equal(
        actual_output, expected_output, check_like=True
    )

  def test_experiment_start_is_aligned_on_start_week_id(
      self,
  ):
    design = experiment_design.ExperimentDesign(
        is_crossover=False,
        runtime_weeks=4,
        pretest_weeks=0,
        **self.irrelevant_design_args
    )

    actual_output = experiment_analysis.add_time_period_column(
        self.data,
        design=design,
        start_week_id=1,
        week_id_column="week_id",
    )

    expected_output = pd.DataFrame({
        "week_id": [-2, -1, 0, 1, 2, 3, 4],
        "time_period": [None, None, None, "test", "test", "test", "test"],
    })
    pd.testing.assert_frame_equal(
        actual_output, expected_output, check_like=True
    )


class PivotTimeAssignmentTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.irrelevant_design_args = dict(
        n_items_before_trimming=1000,
        primary_metric="clicks",
        pre_trim_top_percentile=0.0,
        pre_trim_bottom_percentile=0.0,
        post_trim_percentile=0.1,
    )
    self.data = pd.DataFrame({
        "item_id": [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        "treatment": [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        "week_id": [-2, 0, 1, 2, 2, 3, 4, -1, 1, 1, 5, 7],
        "clicks": [2] * 12,
        "impressions": [3] * 12,
    })
    self.metrics = ["clicks", "impressions"]

  def make_expected_pivot_data(
      self, **week_id_assignments: List[int]
  ) -> pd.DataFrame:
    """Aggregates the data to get the expected metric averages for each period.

    This is a helper for testing to construct the expected data.
    """
    expected_metrics_lst = []
    for period_name, week_ids in week_id_assignments.items():
      expected_metrics = self.data.loc[
          self.data["week_id"].isin(week_ids)
      ].groupby("item_id")[self.metrics].sum() / len(week_ids)
      expected_metrics.columns = pd.MultiIndex.from_product(
          [expected_metrics.columns, [period_name]]
      )
      expected_metrics_lst.append(expected_metrics)

    expected_pivot_data = pd.concat(
        expected_metrics_lst,
        axis=1,
    ).fillna(0)

    expected_column_order = list(
        itertools.product(self.metrics, week_id_assignments.keys())
    )
    return expected_pivot_data[expected_column_order]

  def test_raises_exception_if_time_assignment_column_in_data(
      self,
  ):
    design = experiment_design.ExperimentDesign(
        is_crossover=False,
        runtime_weeks=6,
        crossover_washout_weeks=None,
        pretest_weeks=2,
        **self.irrelevant_design_args
    )
    self.data["time_period"] = "already existing column"

    with self.assertRaisesRegex(
        ValueError,
        "time_period_column must not already be a column in the data.",
    ):
      experiment_analysis.pivot_time_assignment(
          self.data,
          design=design,
          start_week_id=0,
          metric_columns=self.metrics,
          item_id_column="item_id",
          week_id_column="week_id",
      )

  def test_raises_exception_if_metric_column_missing_from_the_data(
      self,
  ):
    design = experiment_design.ExperimentDesign(
        is_crossover=True,
        runtime_weeks=6,
        crossover_washout_weeks=1,
        pretest_weeks=2,
        **self.irrelevant_design_args
    )

    self.data.drop(columns=self.metrics[0], inplace=True)

    with self.assertRaisesRegex(
        ValueError,
        "The dataframe does not have the following columns: {'clicks'}",
    ):
      experiment_analysis.pivot_time_assignment(
          self.data,
          design=design,
          start_week_id=0,
          metric_columns=self.metrics,
          item_id_column="item_id",
          week_id_column="week_id",
      )

  def test_raises_exception_if_item_id_column_missing_from_the_data(
      self,
  ):
    design = experiment_design.ExperimentDesign(
        is_crossover=True,
        runtime_weeks=6,
        crossover_washout_weeks=1,
        pretest_weeks=2,
        **self.irrelevant_design_args
    )

    self.data.drop(columns="item_id", inplace=True)

    with self.assertRaisesRegex(
        ValueError,
        "The dataframe does not have the following columns: {'item_id'}",
    ):
      experiment_analysis.pivot_time_assignment(
          self.data,
          design=design,
          start_week_id=0,
          metric_columns=self.metrics,
          item_id_column="item_id",
          week_id_column="week_id",
      )

  def test_raises_exception_if_week_id_column_missing_from_the_data(
      self,
  ):
    design = experiment_design.ExperimentDesign(
        is_crossover=True,
        runtime_weeks=6,
        crossover_washout_weeks=1,
        pretest_weeks=2,
        **self.irrelevant_design_args
    )

    self.data.drop(columns="week_id", inplace=True)

    with self.assertRaisesRegex(
        ValueError,
        "The dataframe does not have the following columns: {'week_id'}",
    ):
      experiment_analysis.pivot_time_assignment(
          self.data,
          design=design,
          start_week_id=0,
          metric_columns=self.metrics,
          item_id_column="item_id",
          week_id_column="week_id",
      )

  def test_raises_exception_if_treatment_column_specified_and_missing_from_the_data(
      self,
  ):
    design = experiment_design.ExperimentDesign(
        is_crossover=True,
        runtime_weeks=6,
        crossover_washout_weeks=1,
        pretest_weeks=2,
        **self.irrelevant_design_args
    )

    self.data.drop(columns="treatment", inplace=True)

    with self.assertRaisesRegex(
        ValueError,
        "The dataframe does not have the following columns: {'treatment'}",
    ):
      experiment_analysis.pivot_time_assignment(
          self.data,
          design=design,
          start_week_id=0,
          metric_columns=self.metrics,
          item_id_column="item_id",
          week_id_column="week_id",
          treatment_column="treatment",
      )

  def test_pivot_time_assignment_for_regular_experiment(self):
    design = experiment_design.ExperimentDesign(
        is_crossover=False,
        runtime_weeks=6,
        crossover_washout_weeks=None,
        pretest_weeks=2,
        **self.irrelevant_design_args
    )
    # Design:
    # Week -2, -1:  pretest
    # Week 0-5:     test

    pivoted_data = experiment_analysis.pivot_time_assignment(
        self.data,
        design=design,
        start_week_id=0,
        metric_columns=self.metrics,
        item_id_column="item_id",
        week_id_column="week_id",
    )
    expected_pivot_data = self.make_expected_pivot_data(
        pretest=[-2, -1],
        test=[0, 1, 2, 3, 4, 5],
    )

    pd.testing.assert_frame_equal(
        pivoted_data, expected_pivot_data, check_names=False, check_dtype=False
    )

  def test_pivot_time_assignment_for_regular_experiment_without_pretest(self):
    design = experiment_design.ExperimentDesign(
        is_crossover=False,
        runtime_weeks=6,
        crossover_washout_weeks=None,
        pretest_weeks=0,
        **self.irrelevant_design_args
    )
    # Design:
    # Week 0-5:     test

    pivoted_data = experiment_analysis.pivot_time_assignment(
        self.data,
        design=design,
        start_week_id=0,
        metric_columns=self.metrics,
        item_id_column="item_id",
        week_id_column="week_id",
    )
    expected_pivot_data = self.make_expected_pivot_data(
        test=[0, 1, 2, 3, 4, 5],
    )

    pd.testing.assert_frame_equal(
        pivoted_data, expected_pivot_data, check_names=False, check_dtype=False
    )

  def test_pivot_time_assignment_for_regular_experiment_with_shorter_runtime(
      self,
  ):
    design = experiment_design.ExperimentDesign(
        is_crossover=False,
        runtime_weeks=3,
        crossover_washout_weeks=None,
        pretest_weeks=2,
        **self.irrelevant_design_args
    )
    # Design:
    # Week -2, -1:  pretest
    # Week 0-2:     test

    pivoted_data = experiment_analysis.pivot_time_assignment(
        self.data,
        design=design,
        start_week_id=0,
        metric_columns=self.metrics,
        item_id_column="item_id",
        week_id_column="week_id",
    )
    expected_pivot_data = self.make_expected_pivot_data(
        pretest=[-2, -1],
        test=[0, 1, 2],
    )

    pd.testing.assert_frame_equal(
        pivoted_data, expected_pivot_data, check_names=False, check_dtype=False
    )

  def test_pivot_time_assignment_for_crossover_experiment(self):
    design = experiment_design.ExperimentDesign(
        is_crossover=True,
        runtime_weeks=6,
        crossover_washout_weeks=1,
        pretest_weeks=2,
        **self.irrelevant_design_args
    )
    # Design:
    # Week -2, -1:  pretest
    # Week 0:       washout_1
    # Week 1, 2:    test_1
    # Week 3:       washout_2
    # Week 4, 5:    test_2

    pivoted_data = experiment_analysis.pivot_time_assignment(
        self.data,
        design=design,
        start_week_id=0,
        metric_columns=self.metrics,
        item_id_column="item_id",
        week_id_column="week_id",
    )
    expected_pivot_data = self.make_expected_pivot_data(
        pretest=[-2, -1],
        washout_1=[0],
        test_1=[1, 2],
        washout_2=[3],
        test_2=[4, 5],
    )

    pd.testing.assert_frame_equal(
        pivoted_data,
        expected_pivot_data,
        check_names=False,
        check_dtype=False,
        check_like=True,
    )

  def test_pivot_time_assignment_for_crossover_experiment_with_longer_washout(
      self,
  ):
    design = experiment_design.ExperimentDesign(
        is_crossover=True,
        runtime_weeks=6,
        crossover_washout_weeks=2,
        pretest_weeks=2,
        **self.irrelevant_design_args
    )
    # Design:
    # Week -2, -1:  pretest
    # Week 0, 1:    washout_1
    # Week 2:       test_1
    # Week 3, 4:    washout_2
    # Week 5:       test_2

    pivoted_data = experiment_analysis.pivot_time_assignment(
        self.data,
        design=design,
        start_week_id=0,
        metric_columns=self.metrics,
        item_id_column="item_id",
        week_id_column="week_id",
    )
    expected_pivot_data = self.make_expected_pivot_data(
        pretest=[-2, -1],
        washout_1=[0, 1],
        test_1=[2],
        washout_2=[3, 4],
        test_2=[5],
    )

    pd.testing.assert_frame_equal(
        pivoted_data,
        expected_pivot_data,
        check_names=False,
        check_dtype=False,
        check_like=True,
    )

  def test_pivot_time_assignment_for_crossover_experiment_without_pretest(self):
    design = experiment_design.ExperimentDesign(
        is_crossover=True,
        runtime_weeks=6,
        crossover_washout_weeks=1,
        pretest_weeks=0,
        **self.irrelevant_design_args
    )
    # Design:
    # Week 0:       washout_1
    # Week 1, 2:    test_1
    # Week 3:       washout_2
    # Week 4, 5:    test_2

    pivoted_data = experiment_analysis.pivot_time_assignment(
        self.data,
        design=design,
        start_week_id=0,
        metric_columns=self.metrics,
        item_id_column="item_id",
        week_id_column="week_id",
    )
    expected_pivot_data = self.make_expected_pivot_data(
        washout_1=[0],
        test_1=[1, 2],
        washout_2=[3],
        test_2=[4, 5],
    )

    pd.testing.assert_frame_equal(
        pivoted_data,
        expected_pivot_data,
        check_names=False,
        check_dtype=False,
        check_like=True,
    )

  def test_pivot_time_assignment_for_crossover_experiment_with_shorter_runtime(
      self,
  ):
    design = experiment_design.ExperimentDesign(
        is_crossover=True,
        runtime_weeks=4,
        crossover_washout_weeks=1,
        pretest_weeks=2,
        **self.irrelevant_design_args
    )
    # Design:
    # Week -2, -1:  pretest
    # Week 0:       washout_1
    # Week 1:       test_1
    # Week 2:       washout_2
    # Week 3:       test_2

    pivoted_data = experiment_analysis.pivot_time_assignment(
        self.data,
        design=design,
        start_week_id=0,
        metric_columns=self.metrics,
        item_id_column="item_id",
        week_id_column="week_id",
    )
    expected_pivot_data = self.make_expected_pivot_data(
        pretest=[-2, -1],
        washout_1=[0],
        test_1=[1],
        washout_2=[2],
        test_2=[3],
    )

    pd.testing.assert_frame_equal(
        pivoted_data,
        expected_pivot_data,
        check_names=False,
        check_dtype=False,
        check_like=True,
    )

  @parameterized.parameters([True, False])
  def test_pivot_time_assignment_keeps_treatment_assignment_if_treatment_column_not_none(
      self, is_crossover
  ):
    design = experiment_design.ExperimentDesign(
        is_crossover=is_crossover,
        runtime_weeks=4,
        crossover_washout_weeks=1,
        pretest_weeks=2,
        **self.irrelevant_design_args
    )
    # Design:
    # Week -2, -1:  pretest
    # Week 0:       washout
    # Week 1:       test_1
    # Week 2:       washout
    # Week 3:       test_2

    pivoted_data = experiment_analysis.pivot_time_assignment(
        self.data,
        design=design,
        start_week_id=0,
        metric_columns=self.metrics,
        item_id_column="item_id",
        week_id_column="week_id",
        treatment_column="treatment",
    )

    expected_index = [(1, 1), (2, 0)]  # list of (item_id, treatment)
    actual_index = pivoted_data.index.values.tolist()
    self.assertCountEqual(expected_index, actual_index)


class ExperimentAnalysisTests(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    rng = np.random.default_rng(seed=1234)

    self.data = pd.DataFrame({
        "item_id": list(range(100)) * 5,
        "week_id": [-1] * 100 + [0] * 100 + [1] * 100 + [2] * 100 + [3] * 100,
        "clicks": rng.choice(10, size=500).astype(float),
        "impressions": rng.choice(100, size=500).astype(float),
    })
    coinflip = experiment_design.Coinflip(salt="abc123")
    self.data["treatment_assignment"] = self.data["item_id"].apply(coinflip)

    self.crossover_design = experiment_design.ExperimentDesign(
        n_items_before_trimming=100,
        runtime_weeks=4,
        primary_metric="clicks",
        pretest_weeks=1,
        is_crossover=True,
        crossover_washout_weeks=1,
        pre_trim_top_percentile=0.0,
        pre_trim_bottom_percentile=0.0,
        post_trim_percentile=0.05,
        coinflip_salt="abc123",
    )
    self.regular_design = experiment_design.ExperimentDesign(
        n_items_before_trimming=100,
        runtime_weeks=4,
        primary_metric="clicks",
        pretest_weeks=1,
        is_crossover=False,
        pre_trim_top_percentile=0.0,
        pre_trim_bottom_percentile=0.0,
        post_trim_percentile=0.05,
        coinflip_salt="abc123",
    )

  def test_analyze_experiment_returns_expected_dataframe_for_crossover(self):
    metrics = [
        experiment_analysis.Metric(name="Clicks", column="clicks"),
        experiment_analysis.Metric(name="Impressions", column="impressions"),
        experiment_analysis.Metric(
            name="Clicks (no trimm)",
            column="clicks",
            trimming_quantile_override=0.0,
        ),
        experiment_analysis.Metric(
            name="CTR", column="clicks", denominator_column="impressions"
        ),
        experiment_analysis.Metric(
            name="CTR (no trim)",
            column="clicks",
            denominator_column="impressions",
            trimming_quantile_override=0.0,
        ),
    ]

    results = experiment_analysis.analyze_experiment(
        self.data,
        metrics=metrics,
        design=self.crossover_design,
        week_id_column="week_id",
        item_id_column="item_id",
        treatment_assignment_column="treatment_assignment",
    )

    expected_results = pd.DataFrame({
        "metric": [
            "Clicks",
            "Impressions",
            "Clicks (no trimm)",
            "CTR",
            "CTR (no trim)",
        ],
        "is_significant": [False] * 5,
        "alpha": [0.05] * 5,
        "p_value": [
            0.47981624809392687,
            0.6511785126328269,
            0.4342870505283397,
            0.2884113251365828,
            0.34671790805522984,
        ],
        "statistic": [
            -0.7095853125199831,
            0.4536590494716348,
            -0.7850703319578739,
            -1.0679944512388266,
            -0.9454758316151358,
        ],
        "absolute_difference": [
            -0.34444444444444444,
            1.8111111111111111,
            -0.3508000000000001,
            -0.013757147908579084,
            -0.01046454347665346,
        ],
        "absolute_difference_lower_bound": [
            -1.3089567619248998,
            -6.121364178796977,
            -1.2374254120867663,
            -0.039351999854460855,
            -0.032425892314840936,
        ],
        "absolute_difference_upper_bound": [
            0.6200678730360117,
            9.743586401019197,
            0.5358254120867707,
            0.011837704037302686,
            0.011496805361534017,
        ],
        "relative_difference": [
            -0.07524271844660191,
            0.03679458239277644,
            -0.07626086956521705,
            -0.14394918600457385,
            -0.11135639199612779,
        ],
        "relative_difference_lower_bound": [
            -0.25911816160063605,
            -0.11515822156081912,
            -0.2449530881288059,
            -0.3553351642938345,
            -0.30461303788372396,
        ],
        "relative_difference_upper_bound": [
            0.15250922160717417,
            0.21735176439251713,
            0.13013226417039925,
            0.15011624918105748,
            0.14331357952247625,
        ],
        "standard_error": [
            0.4854165360627364,
            3.9922296562153146,
            0.4468389464229858,
            0.012881291557855378,
            0.011068017951106278,
        ],
        "sample_size": [90, 90, 100, 90, 100],
        "degrees_of_freedom": [89, 89, 99, 89, 99],
        "control_average": [
            4.5777777777777775,
            49.22222222222222,
            4.599999999999994,
            0.0955694734400371,
            0.09397344228804883,
        ],
    }).set_index("metric")

    pd.testing.assert_frame_equal(results, expected_results, check_like=True)

  def test_analyze_experiment_returns_expected_dataframe_for_regular_experiment(
      self,
  ):
    metrics = [
        experiment_analysis.Metric(name="Clicks", column="clicks"),
        experiment_analysis.Metric(name="Impressions", column="impressions"),
        experiment_analysis.Metric(
            name="Clicks (no trimm)",
            column="clicks",
            trimming_quantile_override=0.0,
        ),
        experiment_analysis.Metric(
            name="CTR", column="clicks", denominator_column="impressions"
        ),
        experiment_analysis.Metric(
            name="CTR (no trim)",
            column="clicks",
            denominator_column="impressions",
            trimming_quantile_override=0.0,
        ),
    ]

    results = experiment_analysis.analyze_experiment(
        self.data,
        metrics=metrics,
        design=self.regular_design,
        week_id_column="week_id",
        item_id_column="item_id",
        treatment_assignment_column="treatment_assignment",
    )

    expected_results = pd.DataFrame({
        "metric": [
            "Clicks",
            "Impressions",
            "Clicks (no trimm)",
            "CTR",
            "CTR (no trim)",
        ],
        "is_significant": [False] * 5,
        "alpha": [0.05] * 5,
        "p_value": [
            0.5004139975492669,
            0.05707936249907935,
            0.5900129415610891,
            0.1432090186745279,
            0.12240988458432771,
        ],
        "statistic": [
            0.6766612168597068,
            -1.9279922021840314,
            0.5406746796234402,
            1.4771852488068573,
            1.5599513498192001,
        ],
        "absolute_difference": [
            0.18169467836582065,
            -5.54169717070824,
            0.1473638697710875,
            0.011500659455663406,
            0.012378193879969746,
        ],
        "absolute_difference_lower_bound": [
            -0.35200221156510725,
            -11.25379932791624,
            -0.393813755814439,
            -0.0039722230079432316,
            -0.003393947268635872,
        ],
        "absolute_difference_upper_bound": [
            0.7153915682967485,
            0.17040498649976232,
            0.688541495356614,
            0.026973541919270046,
            0.028150335028575363,
        ],
        "relative_difference": [
            0.0409347798602373,
            -0.10609826756026874,
            0.033436977627427034,
            0.13390359386720174,
            0.1463765675896762,
        ],
        "relative_difference_lower_bound": [
            -0.07528562935721528,
            -0.20552673572896996,
            -0.08492448006399,
            -0.042724072352300246,
            -0.03762604639401568,
        ],
        "relative_difference_upper_bound": [
            0.17089717970750828,
            0.003439481543815903,
            0.1654716806256349,
            0.34515897457513267,
            0.3596088110373934,
        ],
        "standard_error": [
            0.26851646560895137,
            2.874335883947352,
            0.27255552243304787,
            0.007785522814388138,
            0.007934987127261622,
        ],
        "sample_size": [92, 92, 100, 92, 100],
        "degrees_of_freedom": [
            87.10024618211024,
            88.04026722874927,
            93.83711686034405,
            87.6780449943031,
            86.80677126875106,
        ],
        "control_average": [
            4.438638218800158,
            52.23174042460492,
            4.4072126199053,
            0.08588760856612362,
            0.08456403974896019,
        ],
    }).set_index("metric")

    pd.testing.assert_frame_equal(results, expected_results, check_like=True)

  @parameterized.product(
      include_relative_effect=[True, False],
      include_absolute_effect=[True, False],
      include_yearly_projected_effect=[True, False],
  )
  def test_format_experiment_analysis_results_dataframe_returns_dataframe_styler(
      self,
      include_relative_effect,
      include_absolute_effect,
      include_yearly_projected_effect,
  ):
    metrics = [
        experiment_analysis.Metric(name="Clicks", column="clicks"),
        experiment_analysis.Metric(name="Impressions", column="impressions"),
        experiment_analysis.Metric(
            name="Clicks (no trimm)",
            column="clicks",
            trimming_quantile_override=0.0,
        ),
        experiment_analysis.Metric(
            name="CTR", column="clicks", denominator_column="impressions"
        ),
        experiment_analysis.Metric(
            name="CTR (no trim)",
            column="clicks",
            denominator_column="impressions",
            trimming_quantile_override=0.0,
        ),
    ]

    results = experiment_analysis.analyze_experiment(
        self.data,
        metrics=metrics,
        design=self.regular_design,
        week_id_column="week_id",
        item_id_column="item_id",
        treatment_assignment_column="treatment_assignment",
    )

    formatted_results = (
        experiment_analysis.format_experiment_analysis_results_dataframe(
            results,
            include_relative_effect=include_relative_effect,
            include_absolute_effect=include_absolute_effect,
            include_yearly_projected_effect=include_yearly_projected_effect,
        )
    )
    shown_columns = formatted_results.columns[[
        n
        for n in range(len(formatted_results.columns))
        if n not in formatted_results.hidden_columns
    ]].values.tolist()

    expected_columns = [
        ("P-value", ""),
    ]
    if include_relative_effect:
      expected_columns += [
          ("Relative Effect Size", "Point Estimate"),
          ("Relative Effect Size", "95% CI"),
      ]
    if include_absolute_effect:
      expected_columns += [
          ("Absolute Effect Size (per item per week)", "Point Estimate"),
          ("Absolute Effect Size (per item per week)", "95% CI"),
      ]
    if include_yearly_projected_effect:
      expected_columns += [
          ("Projected Effect Size (whole feed, per year)", "Point Estimate"),
          ("Projected Effect Size (whole feed, per year)", "95% CI"),
      ]

    self.assertCountEqual(expected_columns, shown_columns)


if __name__ == "__main__":
  absltest.main()
