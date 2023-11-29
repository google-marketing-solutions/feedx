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

"""Tests for experiment_simulations."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pandas as pd

from feedx import experiment_simulations

ExperimentDesign = experiment_simulations.ExperimentDesign


class ExperimentSimulationsTest(parameterized.TestCase):

  def test_generate_all_valid_designs(self):
    actual_designs = experiment_simulations.generate_all_valid_designs(
        max_items_for_test=[500],
        crossover_design_allowed=True,
        traditional_design_allowed=False,
        candidate_runtime_weeks=[6, 8],
        candidate_pretest_weeks=[1],
        candidate_pre_trim_percentiles=[0.0, 0.1],
        candidate_post_trim_percentiles=[0.025],
        primary_metric="clicks",
        crossover_washout_weeks=1,
    )

    expected_designs = [
        ExperimentDesign(
            n_items_before_trimming=500,
            is_crossover=True,
            runtime_weeks=6,
            pretest_weeks=1,
            pre_trim_top_percentile=0.0,
            pre_trim_bottom_percentile=0.0,
            post_trim_percentile=0.025,
            primary_metric="clicks",
            crossover_washout_weeks=1,
        ),
        ExperimentDesign(
            n_items_before_trimming=500,
            is_crossover=True,
            runtime_weeks=6,
            pretest_weeks=1,
            pre_trim_top_percentile=0.1,
            pre_trim_bottom_percentile=0.0,
            post_trim_percentile=0.025,
            primary_metric="clicks",
            crossover_washout_weeks=1,
        ),
        ExperimentDesign(
            n_items_before_trimming=500,
            is_crossover=True,
            runtime_weeks=8,
            pretest_weeks=1,
            pre_trim_top_percentile=0.0,
            pre_trim_bottom_percentile=0.0,
            post_trim_percentile=0.025,
            primary_metric="clicks",
            crossover_washout_weeks=1,
        ),
        ExperimentDesign(
            n_items_before_trimming=500,
            is_crossover=True,
            runtime_weeks=8,
            pretest_weeks=1,
            pre_trim_top_percentile=0.1,
            pre_trim_bottom_percentile=0.0,
            post_trim_percentile=0.025,
            primary_metric="clicks",
            crossover_washout_weeks=1,
        ),
    ]
    self.assertCountEqual(expected_designs, actual_designs)

  def test_does_not_generate_designs_with_pretrimming_and_no_pretest_weeks(
      self,
  ):
    generated_designs = experiment_simulations.generate_all_valid_designs(
        max_items_for_test=[500],
        crossover_design_allowed=True,
        traditional_design_allowed=False,
        candidate_runtime_weeks=[6, 8],
        candidate_pretest_weeks=[0, 1],
        candidate_pre_trim_percentiles=[0.0, 0.1],
        candidate_post_trim_percentiles=[0.025],
        primary_metric='clicks',
        crossover_washout_weeks=1,
    )

    has_pre_trim_but_no_pretest_weeks = [
        (design.pretest_weeks == 0)
        & (
            (design.pre_trim_top_percentile > 0.0)
            | (design.pre_trim_bottom_percentile > 0.0)
        )
        for design in generated_designs
    ]

    self.assertFalse(any(has_pre_trim_but_no_pretest_weeks))

  def test_bootstrap_sample_samples_rows_from_input_data(self):
    input_data = pd.DataFrame(
        {"a": [1, 2, 3, 4, 5, 6], "b": [7, 8, 9, 10, 11, 12]}
    )
    rng = np.random.default_rng(123)
    sampled_data = experiment_simulations._bootstrap_sample_data(
        input_data, rng
    )

    unique_rows_from_sample = set(
        [str(row.values) for _, row in sampled_data.iterrows()]
    )
    unique_rows_from_input = set(
        [str(row.values) for _, row in input_data.iterrows()]
    )
    new_rows = unique_rows_from_sample - unique_rows_from_input
    self.assertEmpty(new_rows)

  def test_bootstrap_sample_samples_input_data_with_replacement(self):
    input_data = pd.DataFrame(
        {"a": [1, 2, 3, 4, 5, 6], "b": [7, 8, 9, 10, 11, 12]}
    )
    rng = np.random.default_rng(123)
    sampled_data = experiment_simulations._bootstrap_sample_data(
        input_data, rng
    )

    # Because we are sampling with replacement, assert that there are some
    # duplicates.
    n_sampled_rows = len(sampled_data.index.values)
    n_unique_sampled_rows = len(
        sampled_data.drop_duplicates(subset=["a", "b"]).index.values
    )
    self.assertLess(n_unique_sampled_rows, n_sampled_rows)

  def test_bootstrap_sample_raises_exception_if_sample_weight_column_exists(
      self,
  ):
    input_data = pd.DataFrame(
        {"a": [1, 2, 3, 4, 5, 6], "b": [7, 8, 9, 10, 11, 12]}
    )
    rng = np.random.default_rng(123)

    with self.assertRaises(ValueError):
      experiment_simulations._bootstrap_sample_data(
          input_data, rng, sample_weight_column="a"
      )

  def test_bootstrap_sample_raises_exception_if_dummy_list_column_exists(self):
    input_data = pd.DataFrame(
        {"a": [1, 2, 3, 4, 5, 6], "b": [7, 8, 9, 10, 11, 12]}
    )
    rng = np.random.default_rng(123)

    with self.assertRaises(ValueError):
      experiment_simulations._bootstrap_sample_data(
          input_data, rng, dummy_list_column="a"
      )


class CalculateMinStartWeekTests(parameterized.TestCase):

  def test_start_week_is_earliest_week_possible_with_max_pretest(self):
    min_start_week_id = experiment_simulations.calculate_minimum_start_week_id(
        candidate_runtime_weeks=[4, 6],
        candidate_pretest_weeks=[0, 2, 4],
        historical_week_ids=range(-6, 4),
    )

    self.assertEqual(min_start_week_id, -2)

  def test_raises_exception_if_not_enough_historical_week_ids(self):
    with self.assertRaises(ValueError):
      # Needs at least 6 runtime weeks and 4 pretest weeks, so 10 weeks total
      # but there are only 9 historical week ids
      experiment_simulations.calculate_minimum_start_week_id(
          candidate_runtime_weeks=[4, 6],
          candidate_pretest_weeks=[0, 2, 4],
          historical_week_ids=range(-6, 3),
      )


class ApplyRandomTreatmentAssignmentTests(parameterized.TestCase):

  def test_apply_random_treatment_assignment_creates_treatment_column(self):
    input_data = pd.DataFrame(
        {"item_id": ["product_1", "product_2", "product_3", "product_4"]}
    )
    rng = np.random.default_rng(seed=123)

    actual_output_data = (
        experiment_simulations.apply_random_treatment_assignment(
            input_data,
            rng=rng,
            item_id_column="item_id",
            treatment_column="treatment_assignment",
        )
    )

    expected_output_data = pd.DataFrame({
        "item_id": ["product_1", "product_2", "product_3", "product_4"],
        "treatment_assignment": [0, 1, 1, 0],
    })
    pd.testing.assert_frame_equal(actual_output_data, expected_output_data)

  def test_apply_random_treatment_assignment_raises_exception_if_treatment_column_exists(
      self,
  ):
    input_data = pd.DataFrame(
        {"item_id": [1, 2, 3], "treatment_assignment": [0, 1, 0]}
    )
    rng = np.random.default_rng(seed=123)

    with self.assertRaisesRegex(
        ValueError,
        "The data already contains a treatment column, cannot add one.",
    ):
      experiment_simulations.apply_random_treatment_assignment(
          input_data,
          rng=rng,
          item_id_column="item_id",
          treatment_column="treatment_assignment",
      )

  def test_apply_random_treatment_assignment_raises_exception_if_item_id_column_does_not_exist(
      self,
  ):
    input_data = pd.DataFrame({"item_id": [1, 2, 3]})
    rng = np.random.default_rng(seed=123)

    with self.assertRaisesRegex(
        ValueError, "The data does not contain the item_id column."
    ):
      experiment_simulations.apply_random_treatment_assignment(
          input_data,
          rng=rng,
          item_id_column="something_else",
          treatment_column="treatment_assignment",
      )


# The experiment design object validates that there are at least 50
# samples for the experiment, to ensure that the statistics will hold.
# We mock this so we can test with small data.
mock_experiment_design_validation = mock.patch.object(
    target=ExperimentDesign, attribute="_validate_design", autospec=True
)


class SimulationAnalysisTests(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    # Example historical data for 3 items across 6 weeks
    # We use primary metric = clicks
    self.historical_data = pd.DataFrame({
        "item_id": [1] * 6 + [2] * 6 + [3] * 6 + [4] * 6 + [5] * 6,
        "week_id": list(range(6)) * 5,
        "clicks": (
            [0, 1, 5, 3, 2, 5]
            + [0, 0, 0, 1, 0, 0]
            + [5, 1, 0, 2, 5, 1]
            + [2, 2, 1, 2, 4, 5]
            + [9, 1, 2, 0, 0, 1]
        ),
    })

  @mock_experiment_design_validation
  def test_after_instantiating_all_analysis_result_attributes_are_none(
      self, mock_validate_design
  ):
    design = ExperimentDesign(
        n_items_before_trimming=5,
        is_crossover=False,
        runtime_weeks=4,
        pretest_weeks=0,
        pre_trim_top_percentile=0.0,
        pre_trim_bottom_percentile=0.0,
        post_trim_percentile=0.2,
        primary_metric="clicks",
        crossover_washout_weeks=1,
    )

    results = experiment_simulations.SimulationAnalysis(
        design=design,
        historical_data=self.historical_data,
        minimum_start_week_id=2,
        week_id_column="week_id",
        item_id_column="item_id",
        rng=np.random.default_rng(0),
    )

    self.assertEqual(
        (
            results.minimum_detectable_effect,
            results.relative_minimum_detectable_effect,
            results.primary_metric_average,
            results.primary_metric_standard_deviation,
            results.aa_simulation_results,
            results.ab_simulation_results,
            results.aa_robustness_pvalue,
            results.ab_robustness_pvalue,
            results.power_at_minimum_detectable_effect,
            results.false_positive_rate,
        ),
        (None, None, None, None, None, None, None, None, None, None),
    )

  @parameterized.named_parameters([
      {
          "testcase_name": "regular_non_cuped",
          "is_crossover": False,
          "pretest_weeks": 0,
          "pre_trim_top_percentile": 0.0,
          "post_trim_percentile": 0.0,
          "expected_minimum_detectable_effect": 5.48073698,
      },
      {
          "testcase_name": "regular_cuped",
          "is_crossover": False,
          "pretest_weeks": 1,
          "pre_trim_top_percentile": 0.0,
          "post_trim_percentile": 0.0,
          "expected_minimum_detectable_effect": 4.11647146,
      },
      {
          "testcase_name": "crossover",
          "is_crossover": True,
          "pretest_weeks": 1,
          "pre_trim_top_percentile": 0.0,
          "post_trim_percentile": 0.0,
          "expected_minimum_detectable_effect": 2.9739277,
      },
      {
          "testcase_name": "with_post_trimming",
          "is_crossover": False,
          "pretest_weeks": 0,
          "pre_trim_top_percentile": 0.0,
          "post_trim_percentile": 0.2,
          "expected_minimum_detectable_effect": 25.9032839,
      },
      {
          "testcase_name": "with_pre_trimming",
          "is_crossover": False,
          "pretest_weeks": 1,
          "pre_trim_top_percentile": 0.2,
          "post_trim_percentile": 0.0,
          "expected_minimum_detectable_effect": 6.5990305,
      },
  ])
  @mock_experiment_design_validation
  def test_estimate_minimum_detectable_effect_gives_expected_minimum_detectable_effect(
      self,
      mock_validate_design,
      is_crossover,
      pretest_weeks,
      pre_trim_top_percentile,
      post_trim_percentile,
      expected_minimum_detectable_effect,
  ):
    design = ExperimentDesign(
        n_items_before_trimming=5,
        is_crossover=is_crossover,
        runtime_weeks=4,
        pretest_weeks=pretest_weeks,
        pre_trim_top_percentile=pre_trim_top_percentile,
        pre_trim_bottom_percentile=0.0,
        post_trim_percentile=post_trim_percentile,
        primary_metric="clicks",
        crossover_washout_weeks=1,
    )

    results = experiment_simulations.SimulationAnalysis(
        design=design,
        historical_data=self.historical_data,
        minimum_start_week_id=2,
        week_id_column="week_id",
        item_id_column="item_id",
        rng=np.random.default_rng(0),
    )
    results.estimate_minimum_detectable_effect()

    self.assertAlmostEqual(
        results.minimum_detectable_effect,
        expected_minimum_detectable_effect,
    )

  @parameterized.named_parameters([
      {
          "testcase_name": "regular_non_cuped",
          "is_crossover": False,
          "pretest_weeks": 0,
          "pre_trim_top_percentile": 0.0,
          "post_trim_percentile": 0.0,
          "expected_relative_minimum_detectable_effect": 2.81063435,
      },
      {
          "testcase_name": "regular_cuped",
          "is_crossover": False,
          "pretest_weeks": 1,
          "pre_trim_top_percentile": 0.0,
          "post_trim_percentile": 0.0,
          "expected_relative_minimum_detectable_effect": 2.111011,
      },
      {
          "testcase_name": "crossover",
          "is_crossover": True,
          "pretest_weeks": 1,
          "pre_trim_top_percentile": 0.0,
          "post_trim_percentile": 0.0,
          "expected_relative_minimum_detectable_effect": 1.48696387,
      },
      {
          "testcase_name": "with_post_trimming",
          "is_crossover": False,
          "pretest_weeks": 0,
          "pre_trim_top_percentile": 0.0,
          "post_trim_percentile": 0.2,
          "expected_relative_minimum_detectable_effect": 13.5147568,
      },
      {
          "testcase_name": "with_pre_trimming",
          "is_crossover": False,
          "pretest_weeks": 1,
          "pre_trim_top_percentile": 0.2,
          "post_trim_percentile": 0.0,
          "expected_relative_minimum_detectable_effect": 3.91053659,
      },
  ])
  @mock_experiment_design_validation
  def test_estimate_minimum_detectable_effect_gives_expected_relative_minimum_detectable_effect(
      self,
      mock_validate_design,
      is_crossover,
      pretest_weeks,
      pre_trim_top_percentile,
      post_trim_percentile,
      expected_relative_minimum_detectable_effect,
  ):
    design = ExperimentDesign(
        n_items_before_trimming=5,
        is_crossover=is_crossover,
        runtime_weeks=4,
        pretest_weeks=pretest_weeks,
        pre_trim_top_percentile=pre_trim_top_percentile,
        pre_trim_bottom_percentile=0.0,
        post_trim_percentile=post_trim_percentile,
        primary_metric="clicks",
        crossover_washout_weeks=1,
    )

    results = experiment_simulations.SimulationAnalysis(
        design=design,
        historical_data=self.historical_data,
        minimum_start_week_id=2,
        week_id_column="week_id",
        item_id_column="item_id",
        rng=np.random.default_rng(0),
    )
    results.estimate_minimum_detectable_effect()

    self.assertAlmostEqual(
        results.relative_minimum_detectable_effect,
        expected_relative_minimum_detectable_effect,
    )

  @parameterized.product(
      is_crossover=[True, False],
      pre_trim_top_percentile=[0.0, 0.2],
      post_trim_percentile=[0.0, 0.2],
  )
  @mock_experiment_design_validation
  def test_relative_minimum_detectable_effect_is_absolute_divided_by_average(
      self,
      mock_validate_design,
      is_crossover,
      pre_trim_top_percentile,
      post_trim_percentile,
  ):
    design = ExperimentDesign(
        n_items_before_trimming=5,
        is_crossover=is_crossover,
        runtime_weeks=4,
        pretest_weeks=1,
        pre_trim_top_percentile=pre_trim_top_percentile,
        pre_trim_bottom_percentile=0.0,
        post_trim_percentile=post_trim_percentile,
        primary_metric="clicks",
        crossover_washout_weeks=1,
    )

    results = experiment_simulations.SimulationAnalysis(
        design=design,
        historical_data=self.historical_data,
        minimum_start_week_id=2,
        week_id_column="week_id",
        item_id_column="item_id",
        rng=np.random.default_rng(0),
    )
    results.estimate_minimum_detectable_effect()

    self.assertAlmostEqual(
        results.minimum_detectable_effect / results.primary_metric_average,
        results.relative_minimum_detectable_effect,
    )


if __name__ == "__main__":
  absltest.main()
