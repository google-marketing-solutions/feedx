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

import itertools
from typing import List
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pandas as pd

from feedx import experiment_analysis
from feedx import experiment_design

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


class ExperimentAnalysisTest(parameterized.TestCase):

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
        "treatment": [0] * 5 + [1] * 5,
        "test": np.linspace(0, 1, 10),
        "pretest": np.ones(10),
    })
    design = experiment_design.ExperimentDesign(
        is_crossover=False,
        post_trim_percentile=0.1,
        pretest_weeks=4,
        **self.irrelevant_design_args
    )

    analysis_result = experiment_analysis.analyze_experiment(data, design)

    self.assertIsInstance(analysis_result, StatisticalTestResults)

  def test_analyze_regular_experiment_uses_cuped_when_pretest_is_provided(self):
    data = pd.DataFrame({
        "treatment": [0] * 5 + [1] * 5,
        "test": np.linspace(0, 1, 10),
        "pretest": [0, 1] * 5,
    })
    design = experiment_design.ExperimentDesign(
        is_crossover=False,
        post_trim_percentile=0.1,
        pretest_weeks=4,
        **self.irrelevant_design_args
    )

    with mock.patch(
        "google3.third_party.professional_services.solutions.feedx.feedx.statistics.apply_cuped_adjustment",
        side_effect=lambda x, *y: x,
    ) as mock_apply_cuped_adjustment:
      experiment_analysis.analyze_experiment(data, design)
      mock_apply_cuped_adjustment.assert_called_once_with(
          data["test"].values,
          data["pretest"].values,
          design.post_trim_percentile,
      )

  def test_analyze_regular_experiment_does_not_use_cuped_when_pretest_is_not_provided(
      self,
  ):
    data = pd.DataFrame({
        "treatment": [0] * 5 + [1] * 5,
        "test": np.linspace(0, 1, 10),
    })
    design = experiment_design.ExperimentDesign(
        is_crossover=False,
        post_trim_percentile=0.1,
        pretest_weeks=0,
        **self.irrelevant_design_args
    )

    with mock.patch(
        "google3.third_party.professional_services.solutions.feedx.feedx.statistics.apply_cuped_adjustment"
    ) as mock_apply_cuped_adjustment:
      experiment_analysis.analyze_experiment(data, design)
      mock_apply_cuped_adjustment.assert_not_called()

  def test_analyze_crossover_experiment_returns_analysis_results(self):
    data = pd.DataFrame({
        "treatment": [0] * 5 + [1] * 5,
        "test_1": np.linspace(0, 1, 10),
        "test_2": np.linspace(0.5, 1.5, 10),
    })
    design = experiment_design.ExperimentDesign(
        is_crossover=True,
        post_trim_percentile=0.1,
        pretest_weeks=4,
        **self.irrelevant_design_args
    )

    analysis_result = experiment_analysis.analyze_experiment(data, design)

    self.assertIsInstance(analysis_result, StatisticalTestResults)

  def test_analyze_crossover_experiment_returns_control_average_when_no_trimming_is_applied(
      self,
  ):
    data = pd.DataFrame({
        "treatment": [0] * 5 + [1] * 5,
        "test_1": np.linspace(0, 1, 10),
        "test_2": np.linspace(0.5, 1.5, 10),
    })
    design = experiment_design.ExperimentDesign(
        is_crossover=True,
        post_trim_percentile=0.0,
        pretest_weeks=4,
        **self.irrelevant_design_args
    )

    analysis_result = experiment_analysis.analyze_experiment(data, design)

    expected_control_average = (
        data.loc[data["treatment"] == 0, "test_1"].sum()
        + data.loc[data["treatment"] == 1, "test_2"].sum()
    ) / len(data.index.values)

    self.assertEqual(analysis_result.control_average, expected_control_average)

  def test_analyze_regular_experiment_returns_control_average_when_no_trimming_is_applied(
      self,
  ):
    data = pd.DataFrame({
        "treatment": [0] * 5 + [1] * 5,
        "test": np.linspace(0, 1, 10),
    })
    design = experiment_design.ExperimentDesign(
        is_crossover=False,
        post_trim_percentile=0.0,
        pretest_weeks=0,
        **self.irrelevant_design_args
    )

    analysis_result = experiment_analysis.analyze_experiment(data, design)

    expected_control_average = data.loc[data["treatment"] == 0, "test"].mean()

    self.assertEqual(analysis_result.control_average, expected_control_average)

  @parameterized.parameters("treatment", "test_1", "test_2")
  def test_raises_exception_if_required_columns_for_crossover_are_missing(
      self, required_missing_column
  ):
    data = pd.DataFrame({
        "treatment": [0] * 5 + [1] * 5,
        "test_1": np.linspace(0, 1, 10),
        "test_2": np.linspace(0.5, 1.5, 10),
    })
    data.drop(columns=required_missing_column, inplace=True)

    design = experiment_design.ExperimentDesign(
        is_crossover=True,
        post_trim_percentile=0.1,
        pretest_weeks=4,
        **self.irrelevant_design_args
    )

    with self.assertRaises(ValueError):
      experiment_analysis.analyze_experiment(data, design)

  @parameterized.parameters("treatment", "test", "pretest")
  def test_raises_exception_if_required_columns_for_regular_experiment_are_missing(
      self, required_missing_column
  ):
    data = pd.DataFrame({
        "treatment": [0] * 5 + [1] * 5,
        "test": np.linspace(0, 1, 10),
        "pretest": np.linspace(0, 1, 10),
    })
    data.drop(columns=required_missing_column, inplace=True)

    design = experiment_design.ExperimentDesign(
        is_crossover=False,
        post_trim_percentile=0.1,
        pretest_weeks=4,
        **self.irrelevant_design_args
    )

    with self.assertRaises(ValueError):
      experiment_analysis.analyze_experiment(data, design)

  def test_pretest_column_not_required_if_pretest_weeks_is_0(self):
    data = pd.DataFrame({
        "treatment": [0] * 5 + [1] * 5,
        "test": np.linspace(0, 1, 10),
    })

    design = experiment_design.ExperimentDesign(
        is_crossover=False,
        post_trim_percentile=0.1,
        pretest_weeks=0,
        **self.irrelevant_design_args
    )

    analysis_results = experiment_analysis.analyze_experiment(data, design)

    self.assertIsInstance(analysis_results, StatisticalTestResults)

  @parameterized.parameters(False, True)
  def test_analyze_experiment_does_not_mutate_original_dataframe(
      self, is_crossover
  ):
    data = pd.DataFrame({
        "treatment": [0] * 5 + [1] * 5,
        "pretest": np.linspace(0, 1, 10),
        "test": np.linspace(0, 1, 10),
        "test_1": np.linspace(0, 1, 10),
        "test_2": np.linspace(0.5, 1.5, 10),
    })
    original_data_copy = data.copy()

    design = experiment_design.ExperimentDesign(
        is_crossover=is_crossover,
        post_trim_percentile=0.1,
        pretest_weeks=4,
        **self.irrelevant_design_args
    )

    experiment_analysis.analyze_experiment(data, design)

    pd.testing.assert_frame_equal(data, original_data_copy)


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


if __name__ == "__main__":
  absltest.main()
