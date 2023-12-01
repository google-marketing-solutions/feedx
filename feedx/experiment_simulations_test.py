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
        primary_metric="clicks",
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
    input_data = pd.DataFrame({
        "a": [1, 2, 3, 4, 5, 6],
        "b": [7, 8, 9, 10, 11, 12],
        "item_id": [1, 2, 3, 4, 5, 6],
    }).set_index("item_id")
    rng = np.random.default_rng(123)
    sampled_data = experiment_simulations._bootstrap_sample_data(
        input_data, rng, item_id_index_name="item_id"
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
    input_data = pd.DataFrame({
        "a": [1, 2, 3, 4, 5, 6],
        "b": [7, 8, 9, 10, 11, 12],
        "item_id": [1, 2, 3, 4, 5, 6],
    }).set_index("item_id")
    rng = np.random.default_rng(123)
    sampled_data = experiment_simulations._bootstrap_sample_data(
        input_data, rng, item_id_index_name="item_id"
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
    input_data = pd.DataFrame({
        "sample_weight": [1, 2, 3, 4, 5, 6],
        "b": [7, 8, 9, 10, 11, 12],
        "item_id": [1, 2, 3, 4, 5, 6],
    }).set_index("item_id")
    rng = np.random.default_rng(123)

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "The input data must not contain the following columns: "
        "{'sample_weight'}",
    ):
      experiment_simulations._bootstrap_sample_data(
          input_data,
          rng,
          sample_weight_column="sample_weight",
          item_id_index_name="item_id",
      )

  def test_bootstrap_sample_raises_exception_if_dummy_list_column_exists(self):
    input_data = pd.DataFrame({
        "dummy_list": [1, 2, 3, 4, 5, 6],
        "b": [7, 8, 9, 10, 11, 12],
        "item_id": [1, 2, 3, 4, 5, 6],
    }).set_index("item_id")
    rng = np.random.default_rng(123)

    with self.assertRaisesWithLiteralMatch(
        ValueError,
        "The input data must not contain the following columns: {'dummy_list'}",
    ):
      experiment_simulations._bootstrap_sample_data(
          input_data,
          rng,
          dummy_list_column="dummy_list",
          item_id_index_name="item_id",
      )

  def test_bootstrap_sample_overwrites_item_ids_with_unique_item_ids(self):
    input_data = pd.DataFrame({
        "a": [1, 2, 3, 4, 5, 6],
        "b": [7, 8, 9, 10, 11, 12],
        "item_id": [6, 7, 8, 9, 10, 11],
    }).set_index("item_id")
    rng = np.random.default_rng(123)

    sampled_data = experiment_simulations._bootstrap_sample_data(
        input_data, rng, item_id_index_name="item_id"
    )

    expected_index = pd.Index(range(6), name="item_id")
    pd.testing.assert_index_equal(sampled_data.index, expected_index)

  def test_bootstrap_sample_overwrites_item_ids_with_unique_item_ids_for_multiindex(
      self,
  ):
    input_data = pd.DataFrame({
        "a": [1, 2, 3, 4, 5, 6],
        "b": [7, 8, 9, 10, 11, 12],
        "item_id": [6, 7, 8, 9, 10, 11],
        "treatment": [0, 0, 0, 1, 1, 1],
    }).set_index(["item_id", "treatment"])
    rng = np.random.default_rng(123)

    sampled_data = experiment_simulations._bootstrap_sample_data(
        input_data, rng, item_id_index_name="item_id"
    )
    expected_index = np.arange(6)
    np.testing.assert_array_equal(
        sampled_data.index.get_level_values("item_id"), expected_index
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


class SyntheticTreatmentEffectTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self.irrelevent_design_args = dict(
        n_items_before_trimming=400,
        runtime_weeks=4,
        pretest_weeks=1,
        pre_trim_top_percentile=0.0,
        pre_trim_bottom_percentile=0.0,
        post_trim_percentile=0.0,
        primary_metric="clicks",
        crossover_washout_weeks=1,
    )

  def test_apply_synthetic_treatment_effect_for_a_regular_experiment(self):
    pivoted_data = pd.DataFrame({
        "item_id": [1, 2, 3, 4],
        "treatment": [0, 1, 0, 1],
        "test": [0.0, 0.1, 0.5, 0.2],
    })
    design = ExperimentDesign(is_crossover=False, **self.irrelevent_design_args)

    actual_data = experiment_simulations.apply_synthetic_treatment_effect(
        pivoted_data,
        design=design,
        effect_size=0.5,
        treatment_column="treatment",
    )
    expected_data = pd.DataFrame({
        "item_id": [1, 2, 3, 4],
        "treatment": [0, 1, 0, 1],
        "test": [0.0, 0.6, 0.5, 0.7],
    })

    pd.testing.assert_frame_equal(actual_data, expected_data)

  def test_apply_synthetic_treatment_effect_for_a_crossover_experiment(self):
    pivoted_data = pd.DataFrame({
        "item_id": [1, 2, 3, 4],
        "treatment": [0, 1, 0, 1],
        "test_1": [0.0, 0.1, 0.5, 0.2],
        "test_2": [0.3, 0.2, 0.9, 0.8],
    })
    design = ExperimentDesign(is_crossover=True, **self.irrelevent_design_args)

    actual_data = experiment_simulations.apply_synthetic_treatment_effect(
        pivoted_data,
        design=design,
        effect_size=0.5,
        treatment_column="treatment",
    )
    expected_data = pd.DataFrame({
        "item_id": [1, 2, 3, 4],
        "treatment": [0, 1, 0, 1],
        "test_1": [0.0, 0.6, 0.5, 0.7],
        "test_2": [0.8, 0.2, 1.4, 0.8],
    })

    pd.testing.assert_frame_equal(actual_data, expected_data)

  @parameterized.parameters(["treatment", "test"])
  def test_apply_synthetic_treatment_effect_raises_if_required_column_missing_for_regular_experiment(
      self, required_column
  ):
    pivoted_data = pd.DataFrame({
        "item_id": [1, 2, 3, 4],
        "treatment": [0, 1, 0, 1],
        "test": [0.0, 0.1, 0.5, 0.2],
    }).drop(required_column, axis=1)
    design = ExperimentDesign(is_crossover=False, **self.irrelevent_design_args)

    with self.assertRaisesRegex(
        ValueError,
        "The pivoted_data is missing the following required columns:"
        f" {{'{required_column}'}}",
    ):
      experiment_simulations.apply_synthetic_treatment_effect(
          pivoted_data,
          design=design,
          effect_size=0.5,
          treatment_column="treatment",
      )

  @parameterized.parameters(["treatment", "test_1", "test_2"])
  def test_apply_synthetic_treatment_effect_raises_if_required_column_missing_for_crossover_experiment(
      self, required_column
  ):
    pivoted_data = pd.DataFrame({
        "item_id": [1, 2, 3, 4],
        "treatment": [0, 1, 0, 1],
        "test_1": [0.0, 0.1, 0.5, 0.2],
        "test_2": [0.3, 0.2, 0.9, 0.8],
    }).drop(required_column, axis=1)
    design = ExperimentDesign(is_crossover=True, **self.irrelevent_design_args)

    with self.assertRaisesRegex(
        ValueError,
        "The pivoted_data is missing the following required columns:"
        f" {{'{required_column}'}}",
    ):
      experiment_simulations.apply_synthetic_treatment_effect(
          pivoted_data,
          design=design,
          effect_size=0.5,
          treatment_column="treatment",
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

    rng = np.random.default_rng(0)
    self.big_historical_data = pd.DataFrame({
        "item_id": list(range(60)) * 6,
        "week_id": sorted(list(range(6)) * 60),
        "clicks": rng.integers(0, 10, 60 * 6),
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
            results.null_p_value_robustness_p_value,
            results.power_robustness_p_value,
            results.simulated_power_at_minimum_detectable_effect,
            results.simulated_false_positive_rate,
            results.false_positive_rate_robustness_p_value,
            results.aa_point_estimate_robustness_p_value,
            results.ab_point_estimate_robustness_p_value,
            results.null_p_value_robustness_check_pass,
            results.power_robustness_check_pass,
            results.false_positive_rate_robustness_check_pass,
            results.aa_point_estimate_robustness_check_pass,
            results.ab_point_estimate_robustness_check_pass,
            results.all_robustness_checks_pass,
        ),
        (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
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

  @mock_experiment_design_validation
  def test_cannot_run_validate_design_before_estimate_minimum_detectable_effect(
      self,
      mock_validate_design,
  ):
    design = ExperimentDesign(
        n_items_before_trimming=5,
        is_crossover=False,
        runtime_weeks=4,
        pretest_weeks=1,
        pre_trim_top_percentile=0.0,
        pre_trim_bottom_percentile=0.0,
        post_trim_percentile=0.0,
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
    with self.assertRaisesWithLiteralMatch(
        RuntimeError,
        "Cannot run validate_design() before"
        " estimate_minimum_detectable_effect().",
    ):
      results.validate_design(n_simulations=5)

  @parameterized.parameters(True, False)
  def test_validate_design_can_be_run_with_or_without_progress_bar(
      self, is_crossover
  ):
    design = ExperimentDesign(
        n_items_before_trimming=60,
        is_crossover=is_crossover,
        runtime_weeks=6,
        pretest_weeks=0,
        pre_trim_top_percentile=0.0,
        pre_trim_bottom_percentile=0.0,
        post_trim_percentile=0.0,
        primary_metric="clicks",
        crossover_washout_weeks=1,
    )

    results_without_pbar = experiment_simulations.SimulationAnalysis(
        design=design,
        historical_data=self.big_historical_data,
        minimum_start_week_id=0,
        week_id_column="week_id",
        item_id_column="item_id",
        rng=np.random.default_rng(0),
    )
    results_without_pbar.estimate_minimum_detectable_effect()
    results_without_pbar.validate_design(n_simulations=5, progress_bar=False)

    results_with_pbar = experiment_simulations.SimulationAnalysis(
        design=design,
        historical_data=self.big_historical_data,
        minimum_start_week_id=0,
        week_id_column="week_id",
        item_id_column="item_id",
        rng=np.random.default_rng(0),
    )
    results_with_pbar.estimate_minimum_detectable_effect()
    results_with_pbar.validate_design(n_simulations=5, progress_bar=True)

    all_simulation_results_with_pbar = pd.concat(
        [
            results_with_pbar.aa_simulation_results,
            results_with_pbar.ab_simulation_results,
        ],
        axis=1,
    )
    all_simulation_results_without_pbar = pd.concat(
        [
            results_without_pbar.aa_simulation_results,
            results_without_pbar.ab_simulation_results,
        ],
        axis=1,
    )

    pd.testing.assert_frame_equal(
        all_simulation_results_with_pbar,
        all_simulation_results_without_pbar,
    )

  def test_validate_design_produces_aa_simulation_results_with_expected_columns(
      self,
  ):
    design = ExperimentDesign(
        n_items_before_trimming=60,
        is_crossover=False,
        runtime_weeks=6,
        pretest_weeks=0,
        pre_trim_top_percentile=0.0,
        pre_trim_bottom_percentile=0.0,
        post_trim_percentile=0.0,
        primary_metric="clicks",
    )

    results = experiment_simulations.SimulationAnalysis(
        design=design,
        historical_data=self.big_historical_data,
        minimum_start_week_id=0,
        week_id_column="week_id",
        item_id_column="item_id",
        rng=np.random.default_rng(0),
    )
    results.estimate_minimum_detectable_effect()
    results.validate_design(n_simulations=5)

    expected_columns = [
        "is_significant",
        "alpha",
        "p_value",
        "statistic",
        "absolute_difference",
        "absolute_difference_lower_bound",
        "absolute_difference_upper_bound",
        "relative_difference",
        "relative_difference_lower_bound",
        "relative_difference_upper_bound",
        "standard_error",
        "sample_size",
        "degrees_of_freedom",
        "control_average",
    ]
    self.assertCountEqual(
        results.aa_simulation_results.columns.values, expected_columns
    )

  def test_validate_design_produces_aa_simulation_results_with_expected_number_of_rows(
      self,
  ):
    design = ExperimentDesign(
        n_items_before_trimming=60,
        is_crossover=False,
        runtime_weeks=6,
        pretest_weeks=0,
        pre_trim_top_percentile=0.0,
        pre_trim_bottom_percentile=0.0,
        post_trim_percentile=0.0,
        primary_metric="clicks",
    )

    results = experiment_simulations.SimulationAnalysis(
        design=design,
        historical_data=self.big_historical_data,
        minimum_start_week_id=0,
        week_id_column="week_id",
        item_id_column="item_id",
        rng=np.random.default_rng(0),
    )
    results.estimate_minimum_detectable_effect()
    results.validate_design(n_simulations=5)

    self.assertLen(results.aa_simulation_results.index.values, 5)

  def test_validate_design_produces_ab_simulation_results_with_expected_columns(
      self,
  ):
    design = ExperimentDesign(
        n_items_before_trimming=60,
        is_crossover=False,
        runtime_weeks=6,
        pretest_weeks=0,
        pre_trim_top_percentile=0.0,
        pre_trim_bottom_percentile=0.0,
        post_trim_percentile=0.0,
        primary_metric="clicks",
    )

    results = experiment_simulations.SimulationAnalysis(
        design=design,
        historical_data=self.big_historical_data,
        minimum_start_week_id=0,
        week_id_column="week_id",
        item_id_column="item_id",
        rng=np.random.default_rng(0),
    )
    results.estimate_minimum_detectable_effect()
    results.validate_design(n_simulations=5)

    expected_columns = [
        "is_significant",
        "alpha",
        "p_value",
        "statistic",
        "absolute_difference",
        "absolute_difference_lower_bound",
        "absolute_difference_upper_bound",
        "relative_difference",
        "relative_difference_lower_bound",
        "relative_difference_upper_bound",
        "standard_error",
        "sample_size",
        "degrees_of_freedom",
        "control_average",
    ]
    self.assertCountEqual(
        results.ab_simulation_results.columns.values, expected_columns
    )

  def test_validate_design_produces_ab_simulation_results_with_expected_number_of_rows(
      self,
  ):
    design = ExperimentDesign(
        n_items_before_trimming=60,
        is_crossover=False,
        runtime_weeks=6,
        pretest_weeks=0,
        pre_trim_top_percentile=0.0,
        pre_trim_bottom_percentile=0.0,
        post_trim_percentile=0.0,
        primary_metric="clicks",
    )

    results = experiment_simulations.SimulationAnalysis(
        design=design,
        historical_data=self.big_historical_data,
        minimum_start_week_id=0,
        week_id_column="week_id",
        item_id_column="item_id",
        rng=np.random.default_rng(0),
    )
    results.estimate_minimum_detectable_effect()
    results.validate_design(n_simulations=5)

    self.assertLen(results.ab_simulation_results.index.values, 5)

  def test_validate_design_p_values_are_different_for_each_aa_simulation(self):
    design = ExperimentDesign(
        n_items_before_trimming=60,
        is_crossover=False,
        runtime_weeks=6,
        pretest_weeks=0,
        pre_trim_top_percentile=0.0,
        pre_trim_bottom_percentile=0.0,
        post_trim_percentile=0.0,
        primary_metric="clicks",
    )

    results = experiment_simulations.SimulationAnalysis(
        design=design,
        historical_data=self.big_historical_data,
        minimum_start_week_id=0,
        week_id_column="week_id",
        item_id_column="item_id",
        rng=np.random.default_rng(0),
    )
    results.estimate_minimum_detectable_effect()
    results.validate_design(n_simulations=5)

    first_p_value = results.aa_simulation_results.loc[0, "p_value"]
    self.assertFalse(
        np.all(results.aa_simulation_results["p_value"] == first_p_value)
    )

  def test_validate_design_p_values_are_different_for_each_ab_simulation(self):
    design = ExperimentDesign(
        n_items_before_trimming=60,
        is_crossover=False,
        runtime_weeks=6,
        pretest_weeks=0,
        pre_trim_top_percentile=0.0,
        pre_trim_bottom_percentile=0.0,
        post_trim_percentile=0.0,
        primary_metric="clicks",
    )

    results = experiment_simulations.SimulationAnalysis(
        design=design,
        historical_data=self.big_historical_data,
        minimum_start_week_id=0,
        week_id_column="week_id",
        item_id_column="item_id",
        rng=np.random.default_rng(0),
    )
    results.estimate_minimum_detectable_effect()
    results.validate_design(n_simulations=5)

    first_p_value = results.aa_simulation_results.loc[0, "p_value"]
    self.assertFalse(
        np.all(results.ab_simulation_results["p_value"] == first_p_value)
    )

  def test_null_p_value_robustness_p_value_returns_expected_value(self):
    design = ExperimentDesign(
        n_items_before_trimming=60,
        is_crossover=False,
        runtime_weeks=6,
        pretest_weeks=0,
        pre_trim_top_percentile=0.0,
        pre_trim_bottom_percentile=0.0,
        post_trim_percentile=0.0,
        primary_metric="clicks",
    )

    results = experiment_simulations.SimulationAnalysis(
        design=design,
        historical_data=self.big_historical_data,
        minimum_start_week_id=0,
        week_id_column="week_id",
        item_id_column="item_id",
        rng=np.random.default_rng(0),
    )
    results.estimate_minimum_detectable_effect()
    results.validate_design(n_simulations=5)

    # Override the relevant column to isolate this test
    results.aa_simulation_results["p_value"] = np.array(
        [0.1, 0.2, 0.5, 0.3, 0.8]
    )

    self.assertAlmostEqual(results.null_p_value_robustness_p_value, 0.664)

  def test_simulated_false_positive_rate_returns_expected_value(self):
    design = ExperimentDesign(
        n_items_before_trimming=60,
        is_crossover=False,
        runtime_weeks=6,
        pretest_weeks=0,
        pre_trim_top_percentile=0.0,
        pre_trim_bottom_percentile=0.0,
        post_trim_percentile=0.0,
        primary_metric="clicks",
    )

    results = experiment_simulations.SimulationAnalysis(
        design=design,
        historical_data=self.big_historical_data,
        minimum_start_week_id=0,
        week_id_column="week_id",
        item_id_column="item_id",
        rng=np.random.default_rng(0),
    )
    results.estimate_minimum_detectable_effect()
    results.validate_design(n_simulations=5)

    # Override the relevant column to isolate this test
    results.aa_simulation_results["p_value"] = np.array(
        [0.01, 0.2, 0.5, 0.3, 0.8]
    )

    self.assertAlmostEqual(results.simulated_false_positive_rate, 0.2)

  def test_false_positive_rate_robustness_p_value_returns_expected_value(self):
    design = ExperimentDesign(
        n_items_before_trimming=60,
        is_crossover=False,
        runtime_weeks=6,
        pretest_weeks=0,
        pre_trim_top_percentile=0.0,
        pre_trim_bottom_percentile=0.0,
        post_trim_percentile=0.0,
        primary_metric="clicks",
    )

    results = experiment_simulations.SimulationAnalysis(
        design=design,
        historical_data=self.big_historical_data,
        minimum_start_week_id=0,
        week_id_column="week_id",
        item_id_column="item_id",
        rng=np.random.default_rng(0),
    )
    results.estimate_minimum_detectable_effect()
    results.validate_design(n_simulations=5)

    # Override the relevant column to isolate this test
    results.aa_simulation_results["p_value"] = np.array(
        [0.01, 0.2, 0.5, 0.3, 0.8]
    )

    self.assertAlmostEqual(
        results.false_positive_rate_robustness_p_value, 0.22621906
    )

  def test_simulated_power_at_minimum_detectable_effect_returns_expected_value(
      self,
  ):
    design = ExperimentDesign(
        n_items_before_trimming=60,
        is_crossover=False,
        runtime_weeks=6,
        pretest_weeks=0,
        pre_trim_top_percentile=0.0,
        pre_trim_bottom_percentile=0.0,
        post_trim_percentile=0.0,
        primary_metric="clicks",
    )

    results = experiment_simulations.SimulationAnalysis(
        design=design,
        historical_data=self.big_historical_data,
        minimum_start_week_id=0,
        week_id_column="week_id",
        item_id_column="item_id",
        rng=np.random.default_rng(0),
    )
    results.estimate_minimum_detectable_effect()
    results.validate_design(n_simulations=5)

    # Override the relevant column to isolate this test
    results.ab_simulation_results["p_value"] = np.array(
        [0.01, 0.01, 0.01, 0.3, 0.8]
    )

    self.assertAlmostEqual(
        results.simulated_power_at_minimum_detectable_effect, 0.6
    )

  def test_power_robustness_p_value_returns_expected_value(
      self,
  ):
    design = ExperimentDesign(
        n_items_before_trimming=60,
        is_crossover=False,
        runtime_weeks=6,
        pretest_weeks=0,
        pre_trim_top_percentile=0.0,
        pre_trim_bottom_percentile=0.0,
        post_trim_percentile=0.0,
        primary_metric="clicks",
    )

    results = experiment_simulations.SimulationAnalysis(
        design=design,
        historical_data=self.big_historical_data,
        minimum_start_week_id=0,
        week_id_column="week_id",
        item_id_column="item_id",
        rng=np.random.default_rng(0),
    )
    results.estimate_minimum_detectable_effect()
    results.validate_design(n_simulations=5)

    # Override the relevant column to isolate this test
    results.ab_simulation_results["p_value"] = np.array(
        [0.01, 0.01, 0.01, 0.3, 0.8]
    )

    self.assertAlmostEqual(results.power_robustness_p_value, 0.26272)

  def test_aa_point_estimate_robustness_p_value_returns_expected_value(
      self,
  ):
    design = ExperimentDesign(
        n_items_before_trimming=60,
        is_crossover=False,
        runtime_weeks=6,
        pretest_weeks=0,
        pre_trim_top_percentile=0.0,
        pre_trim_bottom_percentile=0.0,
        post_trim_percentile=0.0,
        primary_metric="clicks",
    )

    results = experiment_simulations.SimulationAnalysis(
        design=design,
        historical_data=self.big_historical_data,
        minimum_start_week_id=0,
        week_id_column="week_id",
        item_id_column="item_id",
        rng=np.random.default_rng(0),
    )
    results.estimate_minimum_detectable_effect()
    results.validate_design(n_simulations=5)

    # Override the relevant column to isolate this test
    results.aa_simulation_results["absolute_difference"] = np.array(
        [0.01, -0.03, 0.5, -0.3, 0.71]
    )

    self.assertAlmostEqual(
        results.aa_point_estimate_robustness_p_value, 0.39113626
    )

  def test_ab_point_estimate_robustness_p_value_returns_expected_value(
      self,
  ):
    design = ExperimentDesign(
        n_items_before_trimming=60,
        is_crossover=False,
        runtime_weeks=6,
        pretest_weeks=0,
        pre_trim_top_percentile=0.0,
        pre_trim_bottom_percentile=0.0,
        post_trim_percentile=0.0,
        primary_metric="clicks",
    )

    results = experiment_simulations.SimulationAnalysis(
        design=design,
        historical_data=self.big_historical_data,
        minimum_start_week_id=0,
        week_id_column="week_id",
        item_id_column="item_id",
        rng=np.random.default_rng(0),
    )
    results.estimate_minimum_detectable_effect()
    results.validate_design(n_simulations=5)

    # Override the relevant column to isolate this test
    results.ab_simulation_results["absolute_difference"] = np.array(
        [0.01, -0.03, 0.5, -0.3, 0.71]
    )

    self.assertAlmostEqual(
        results.ab_point_estimate_robustness_p_value, 0.0152695
    )

  @parameterized.parameters(
      "null_p_value",
      "power",
      "false_positive_rate",
      "ab_point_estimate",
      "aa_point_estimate",
  )
  def test_robustness_checks_return_false_if_pvalue_smaller_than_threshold(
      self, robustness_check
  ):
    design = ExperimentDesign(
        n_items_before_trimming=60,
        is_crossover=False,
        runtime_weeks=6,
        pretest_weeks=0,
        pre_trim_top_percentile=0.0,
        pre_trim_bottom_percentile=0.0,
        post_trim_percentile=0.0,
        primary_metric="clicks",
    )

    with mock.patch.object(
        experiment_simulations.SimulationAnalysis,
        f"{robustness_check}_robustness_p_value",
        new_callable=mock.PropertyMock,
    ) as mock_p_value:
      mock_p_value.return_value = 0.00001
      results = experiment_simulations.SimulationAnalysis(
          design=design,
          historical_data=self.big_historical_data,
          minimum_start_week_id=0,
          week_id_column="week_id",
          item_id_column="item_id",
          rng=np.random.default_rng(0),
      )
      self.assertFalse(
          getattr(results, f"{robustness_check}_robustness_check_pass")
      )

  @parameterized.parameters(
      "null_p_value",
      "power",
      "false_positive_rate",
      "ab_point_estimate",
      "aa_point_estimate",
  )
  def test_robustness_checks_return_true_if_pvalue_higher_than_threshold(
      self, robustness_check
  ):
    design = ExperimentDesign(
        n_items_before_trimming=60,
        is_crossover=False,
        runtime_weeks=6,
        pretest_weeks=0,
        pre_trim_top_percentile=0.0,
        pre_trim_bottom_percentile=0.0,
        post_trim_percentile=0.0,
        primary_metric="clicks",
    )

    with mock.patch.object(
        experiment_simulations.SimulationAnalysis,
        f"{robustness_check}_robustness_p_value",
        new_callable=mock.PropertyMock,
    ) as mock_p_value:
      mock_p_value.return_value = 0.99999
      results = experiment_simulations.SimulationAnalysis(
          design=design,
          historical_data=self.big_historical_data,
          minimum_start_week_id=0,
          week_id_column="week_id",
          item_id_column="item_id",
          rng=np.random.default_rng(0),
      )
      self.assertTrue(
          getattr(results, f"{robustness_check}_robustness_check_pass")
      )

  def test_all_robustness_checks_pass_returns_true_if_all_are_true(self):
    design = ExperimentDesign(
        n_items_before_trimming=60,
        is_crossover=False,
        runtime_weeks=6,
        pretest_weeks=0,
        pre_trim_top_percentile=0.0,
        pre_trim_bottom_percentile=0.0,
        post_trim_percentile=0.0,
        primary_metric="clicks",
    )

    with mock.patch.object(
        experiment_simulations.SimulationAnalysis,
        "_is_above_p_value_threshold",
    ) as mock_is_above_p_value_threshold:
      mock_is_above_p_value_threshold.return_value = True
      results = experiment_simulations.SimulationAnalysis(
          design=design,
          historical_data=self.big_historical_data,
          minimum_start_week_id=0,
          week_id_column="week_id",
          item_id_column="item_id",
          rng=np.random.default_rng(0),
      )
      self.assertTrue(results.all_robustness_checks_pass)

  @parameterized.parameters(
      "null_p_value",
      "power",
      "false_positive_rate",
      "ab_point_estimate",
      "aa_point_estimate",
  )
  def test_all_robustness_checks_pass_returns_false_if_any_are_false(
      self, failing_robustness_check
  ):
    design = ExperimentDesign(
        n_items_before_trimming=60,
        is_crossover=False,
        runtime_weeks=6,
        pretest_weeks=0,
        pre_trim_top_percentile=0.0,
        pre_trim_bottom_percentile=0.0,
        post_trim_percentile=0.0,
        primary_metric="clicks",
    )

    with mock.patch.object(
        experiment_simulations.SimulationAnalysis,
        "_is_above_p_value_threshold",
    ) as mock_is_above_p_value_threshold:
      mock_is_above_p_value_threshold.return_value = True
      with mock.patch.object(
          experiment_simulations.SimulationAnalysis,
          f"{failing_robustness_check}_robustness_check_pass",
          new_callable=mock.PropertyMock,
      ) as mock_failing_check:
        mock_failing_check.return_value = False
        results = experiment_simulations.SimulationAnalysis(
            design=design,
            historical_data=self.big_historical_data,
            minimum_start_week_id=0,
            week_id_column="week_id",
            item_id_column="item_id",
            rng=np.random.default_rng(0),
        )
        self.assertFalse(results.all_robustness_checks_pass)

  def test_summay_dict_returns_dict_with_parameters_and_results(self):
    design = ExperimentDesign(
        n_items_before_trimming=60,
        is_crossover=False,
        runtime_weeks=6,
        pretest_weeks=0,
        pre_trim_top_percentile=0.0,
        pre_trim_bottom_percentile=0.0,
        post_trim_percentile=0.0,
        primary_metric="clicks",
    )

    results = experiment_simulations.SimulationAnalysis(
        design=design,
        historical_data=self.big_historical_data,
        minimum_start_week_id=0,
        week_id_column="week_id",
        item_id_column="item_id",
        rng=np.random.default_rng(0),
        robustness_p_value_threshold=0.1,
    )
    results.estimate_minimum_detectable_effect()
    results.validate_design(n_simulations=5)

    actual_summary_dict = results.summary_dict()
    expected_summary_dict = {
        "design_id": design.design_id,
        "aa_point_estimate_robustness_check_pass": (
            results.aa_point_estimate_robustness_check_pass
        ),
        "ab_point_estimate_robustness_check_pass": (
            results.ab_point_estimate_robustness_check_pass
        ),
        "alpha": design.alpha,
        "crossover_washout_weeks": design.crossover_washout_weeks,
        "false_positive_rate_robustness_check_pass": (
            results.false_positive_rate_robustness_check_pass
        ),
        "is_crossover": design.is_crossover,
        "minimum_detectable_effect": results.minimum_detectable_effect,
        "n_items_after_post_trim": design.n_items_after_post_trim,
        "n_items_after_pre_trim": design.n_items_after_pre_trim,
        "n_items_before_trimming": design.n_items_before_trimming,
        "null_p_value_robustness_check_pass": (
            results.null_p_value_robustness_check_pass
        ),
        "post_trim_percentile": design.post_trim_percentile,
        "power": design.power,
        "power_robustness_check_pass": results.power_robustness_check_pass,
        "pre_trim_bottom_percentile": design.pre_trim_bottom_percentile,
        "pre_trim_top_percentile": design.pre_trim_top_percentile,
        "pretest_weeks": design.pretest_weeks,
        "primary_metric": design.primary_metric,
        "relative_minimum_detectable_effect": (
            results.relative_minimum_detectable_effect
        ),
        "runtime_weeks": design.runtime_weeks,
        "simulated_false_positive_rate": results.simulated_false_positive_rate,
        "simulated_power_at_minimum_detectable_effect": (
            results.simulated_power_at_minimum_detectable_effect
        ),
        "all_robustness_checks_pass": results.all_robustness_checks_pass,
    }

    self.assertDictEqual(actual_summary_dict, expected_summary_dict)

  def test_summay_dict_results_are_none_if_not_calculated(self):
    design = ExperimentDesign(
        n_items_before_trimming=60,
        is_crossover=False,
        runtime_weeks=6,
        pretest_weeks=0,
        pre_trim_top_percentile=0.0,
        pre_trim_bottom_percentile=0.0,
        post_trim_percentile=0.0,
        primary_metric="clicks",
    )

    results = experiment_simulations.SimulationAnalysis(
        design=design,
        historical_data=self.big_historical_data,
        minimum_start_week_id=0,
        week_id_column="week_id",
        item_id_column="item_id",
        rng=np.random.default_rng(0),
        robustness_p_value_threshold=0.1,
    )

    actual_summary_dict = results.summary_dict()
    expected_summary_dict = {
        "design_id": design.design_id,
        "aa_point_estimate_robustness_check_pass": None,
        "ab_point_estimate_robustness_check_pass": None,
        "alpha": design.alpha,
        "crossover_washout_weeks": design.crossover_washout_weeks,
        "false_positive_rate_robustness_check_pass": None,
        "is_crossover": design.is_crossover,
        "minimum_detectable_effect": None,
        "n_items_after_post_trim": design.n_items_after_post_trim,
        "n_items_after_pre_trim": design.n_items_after_pre_trim,
        "n_items_before_trimming": design.n_items_before_trimming,
        "null_p_value_robustness_check_pass": None,
        "post_trim_percentile": design.post_trim_percentile,
        "power": design.power,
        "power_robustness_check_pass": None,
        "pre_trim_bottom_percentile": design.pre_trim_bottom_percentile,
        "pre_trim_top_percentile": design.pre_trim_top_percentile,
        "pretest_weeks": design.pretest_weeks,
        "primary_metric": design.primary_metric,
        "relative_minimum_detectable_effect": None,
        "runtime_weeks": design.runtime_weeks,
        "simulated_false_positive_rate": None,
        "simulated_power_at_minimum_detectable_effect": None,
        "all_robustness_checks_pass": None,
    }

    self.assertDictEqual(actual_summary_dict, expected_summary_dict)


if __name__ == "__main__":
  absltest.main()
