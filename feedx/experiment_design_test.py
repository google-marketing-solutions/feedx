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

"""Tests for experiment_design."""

import textwrap

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pandas as pd

from feedx import experiment_design


ExperimentDesign = experiment_design.ExperimentDesign


class ExperimentDesignTest(parameterized.TestCase):

  def test_design_calculates_n_items_after_pre_trim_correctly(self):
    design = ExperimentDesign(
        n_items_before_trimming=1000,
        runtime_weeks=6,
        primary_metric="clicks",
        pretest_weeks=4,
        is_crossover=True,
        pre_trim_top_percentile=0.1,
        pre_trim_bottom_percentile=0.3,
        post_trim_percentile=0.2,
        crossover_washout_weeks=2,
        alpha=0.05,
        power=0.8,
    )

    self.assertEqual(design.n_items_after_pre_trim, 600)

  def test_design_calculates_n_items_after_post_trim_correctly(self):
    design = ExperimentDesign(
        n_items_before_trimming=1000,
        runtime_weeks=6,
        primary_metric="clicks",
        pretest_weeks=4,
        is_crossover=True,
        pre_trim_top_percentile=0.1,
        pre_trim_bottom_percentile=0.3,
        post_trim_percentile=0.2,
        crossover_washout_weeks=2,
        alpha=0.05,
        power=0.8,
    )

    self.assertEqual(design.n_items_after_post_trim, 360)

  @parameterized.named_parameters([
      {
          "testcase_name": "Too few items",
          "invalid_args": {"n_items_before_trimming": 49},
      },
      {
          "testcase_name": "Too few items after trimming",
          "invalid_args": {"pre_trim_top_percentile": 0.95},
      },
      {
          "testcase_name": "Zero runtime weeks",
          "invalid_args": {"runtime_weeks": 0},
      },
      {
          "testcase_name": "Negative pretest weeks",
          "invalid_args": {"pretest_weeks": -1},
      },
      {
          "testcase_name": "Negative post_trim_percentile",
          "invalid_args": {"post_trim_percentile": -0.1},
      },
      {
          "testcase_name": "Post trim percentile 0.5 or more",
          "invalid_args": {"post_trim_percentile": 0.5},
      },
      {
          "testcase_name": "Negative pre_trim_top_percentile",
          "invalid_args": {"pre_trim_top_percentile": -0.1},
      },
      {
          "testcase_name": "Negative pre_trim_bottom_percentile",
          "invalid_args": {"pre_trim_bottom_percentile": -0.1},
      },
      {
          "testcase_name": "Pre trim percentiles sum >= 1.0",
          "invalid_args": {
              "pre_trim_top_percentile": 0.4,
              "pre_trim_bottom_percentile": 0.6,
          },
      },
      {
          "testcase_name": (
              "Missing crossover washout weeks for crossover design"
          ),
          "invalid_args": {
              "crossover_washout_weeks": None,
              "is_crossover": True,
          },
      },
      {
          "testcase_name": (
              "Negative crossover washout weeks for crossover design"
          ),
          "invalid_args": {
              "crossover_washout_weeks": -1,
              "is_crossover": True,
          },
      },
      {
          "testcase_name": "Runtime weeks less than 2*crossover washout weeks",
          "invalid_args": {
              "crossover_washout_weeks": 3,
              "runtime_weeks": 4,
              "is_crossover": True,
          },
      },
      {
          "testcase_name": "Odd runtime weeks for crossover design",
          "invalid_args": {"runtime_weeks": 3, "is_crossover": True},
      },
      {
          "testcase_name": "Pretest trimming without pretest weeks",
          "invalid_args": {"pretest_weeks": 0, "pre_trim_top_percentile": 0.1},
      },
  ])
  def test_raises_value_error_if_input_args_are_invalid(self, invalid_args):
    design_args = dict(
        n_items_before_trimming=100,
        runtime_weeks=6,
        primary_metric="clicks",
        pretest_weeks=4,
        is_crossover=True,
        pre_trim_top_percentile=0.0,
        pre_trim_bottom_percentile=0.0,
        post_trim_percentile=0.0,
        crossover_washout_weeks=1,
        alpha=0.05,
        power=0.8,
    )
    design_args.update(invalid_args)

    with self.assertRaises(ValueError):
      ExperimentDesign(**design_args)

  def test_design_id_is_different_if_args_are_different(self):
    design_args = dict(
        n_items_before_trimming=1000,
        runtime_weeks=6,
        primary_metric="clicks",
        pretest_weeks=4,
        is_crossover=True,
        pre_trim_top_percentile=0.1,
        pre_trim_bottom_percentile=0.3,
        post_trim_percentile=0.2,
        crossover_washout_weeks=2,
        alpha=0.05,
        power=0.8,
    )

    changed_design_args = design_args.copy()
    changed_design_args["runtime_weeks"] = 10

    design = ExperimentDesign(**design_args)
    changed_design = ExperimentDesign(**changed_design_args)

    self.assertNotEqual(design.design_id, changed_design.design_id)

  def test_design_id_is_same_if_args_are_same(self):
    design_args = dict(
        n_items_before_trimming=1000,
        runtime_weeks=6,
        primary_metric="clicks",
        pretest_weeks=4,
        is_crossover=True,
        pre_trim_top_percentile=0.1,
        pre_trim_bottom_percentile=0.3,
        post_trim_percentile=0.2,
        crossover_washout_weeks=2,
        alpha=0.05,
        power=0.8,
    )

    design = ExperimentDesign(**design_args)
    unchanged_design = ExperimentDesign(**design_args)

    self.assertEqual(design.design_id, unchanged_design.design_id)

  def test_design_can_be_written_to_yaml(self):
    design = ExperimentDesign(
        n_items_before_trimming=1000,
        runtime_weeks=6,
        primary_metric="clicks",
        pretest_weeks=4,
        is_crossover=True,
        pre_trim_top_percentile=0.1,
        pre_trim_bottom_percentile=0.3,
        post_trim_percentile=0.2,
        crossover_washout_weeks=2,
        alpha=0.05,
        power=0.8,
    )

    yaml_file = self.create_tempfile()
    design.write_to_yaml(yaml_file.full_path)

    expected_yaml = textwrap.dedent("""\
        alpha: 0.05
        crossover_washout_weeks: 2
        is_crossover: true
        n_items_after_post_trim: 360
        n_items_after_pre_trim: 600
        n_items_before_trimming: 1000
        post_trim_percentile: 0.2
        power: 0.8
        pre_trim_bottom_percentile: 0.3
        pre_trim_top_percentile: 0.1
        pretest_weeks: 4
        primary_metric: clicks
        runtime_weeks: 6
    """)
    self.assertEqual(yaml_file.read_text(), expected_yaml)

  def test_design_can_be_loaded_from_yaml(self):
    design = ExperimentDesign(
        n_items_before_trimming=1000,
        runtime_weeks=6,
        primary_metric="clicks",
        pretest_weeks=4,
        is_crossover=True,
        pre_trim_top_percentile=0.1,
        pre_trim_bottom_percentile=0.3,
        post_trim_percentile=0.2,
        crossover_washout_weeks=2,
        alpha=0.05,
        power=0.8,
    )

    yaml_file = self.create_tempfile()
    design.write_to_yaml(yaml_file.full_path)
    loaded_design = ExperimentDesign.load_from_yaml(yaml_file.full_path)

    self.assertEqual(loaded_design, design)

  def test_raises_value_error_if_n_items_after_post_trim_not_correct_in_yaml(
      self,
  ):
    yaml_file = self.create_tempfile()
    yaml_file.write_text(textwrap.dedent("""\
        alpha: 0.05
        crossover_washout_weeks: 2
        is_crossover: true
        n_items_after_post_trim: 200
        n_items_after_pre_trim: 600
        n_items_before_trimming: 1000
        post_trim_percentile: 0.2
        power: 0.8
        pre_trim_bottom_percentile: 0.3
        pre_trim_top_percentile: 0.1
        pretest_weeks: 4
        primary_metric: clicks
        runtime_weeks: 6
    """))

    with self.assertRaises(ValueError):
      ExperimentDesign.load_from_yaml(yaml_file.full_path)

  def test_raises_value_error_if_n_items_after_pre_trim_not_correct_in_yaml(
      self,
  ):
    yaml_file = self.create_tempfile()
    yaml_file.write_text(textwrap.dedent("""\
        alpha: 0.05
        crossover_washout_weeks: 2
        is_crossover: true
        n_items_after_post_trim: 360
        n_items_after_pre_trim: 500
        n_items_before_trimming: 1000
        post_trim_percentile: 0.2
        power: 0.8
        pre_trim_bottom_percentile: 0.3
        pre_trim_top_percentile: 0.1
        pretest_weeks: 4
        primary_metric: clicks
        runtime_weeks: 6
    """))

    with self.assertRaises(ValueError):
      ExperimentDesign.load_from_yaml(yaml_file.full_path)


class CoinflipTests(parameterized.TestCase):

  def test_coinflip_is_always_0_or_1(self):
    item_ids = [f"Product_{i}" for i in range(100)]

    coinflip = experiment_design.Coinflip(salt="abc")
    coinflip_unique_output = set(map(coinflip, item_ids))

    self.assertEqual(coinflip_unique_output, {0, 1})

  def test_coinflip_is_deterministic_for_item_id_with_same_salt(self):
    item_ids = ["Product_1"] * 100

    coinflip = experiment_design.Coinflip(salt="abc")
    coinflip_unique_output = set(map(coinflip, item_ids))

    self.assertLen(coinflip_unique_output, 1)

  def test_different_coinflip_instances_produces_same_output_if_salt_is_same(
      self,
  ):
    coinflip_1 = experiment_design.Coinflip(salt="abc")
    coinflip_2 = experiment_design.Coinflip(salt="abc")
    item_ids = [f"Product_{i}" for i in range(100)]

    coinflip_1_outputs = list(map(coinflip_1, item_ids))
    coinflip_2_outputs = list(map(coinflip_2, item_ids))

    self.assertListEqual(coinflip_1_outputs, coinflip_2_outputs)

  def test_coinflip_is_different_for_different_salts(self):
    item_id = "Product_1"

    coinflip_unique_output = set(
        [
            experiment_design.Coinflip(salt=str(salt))(item_id)
            for salt in range(100)
        ]
    )

    self.assertLen(coinflip_unique_output, 2)

  def test_coinfip_with_random_salt_generates_a_random_salt(self):
    coinflip_1 = experiment_design.Coinflip.with_random_salt()
    coinflip_2 = experiment_design.Coinflip.with_random_salt()

    self.assertNotEqual(coinflip_1.salt, coinflip_2.salt)

  def test_coinfip_with_random_salt_produces_different_treatment_assignments(
      self,
  ):
    coinflip_1 = experiment_design.Coinflip.with_random_salt()
    coinflip_2 = experiment_design.Coinflip.with_random_salt()
    item_ids = [f"Product_{i}" for i in range(100)]

    coinflip_1_outputs = list(map(coinflip_1, item_ids))
    coinflip_2_outputs = list(map(coinflip_2, item_ids))

    self.assertNotEqual(coinflip_1_outputs, coinflip_2_outputs)

  def test_salt_cannot_be_empty_string(self):
    with self.assertRaises(ValueError):
      experiment_design.Coinflip(salt="")


class TrimOutliersTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.data = pd.DataFrame(
        {"a": [10, 4, 1, 6, 3, 4, 2, 2, 5, 7], "b": ["something"] * 10}
    )
    self.data["a_rank"] = self.data["a"].rank(method="first")
    self.rng = np.random.default_rng(123)

  def test_if_percentiles_are_zero_no_outliers_are_trimmed(self):
    output_data = experiment_design.trim_outliers(
        data=self.data,
        order_by="a",
        trim_percentile_top=0.0,
        trim_percentile_bottom=0.0,
        rng=self.rng,
    )
    pd.testing.assert_frame_equal(output_data, self.data)

  def test_if_percentiles_represent_less_than_1_row_no_outliers_are_trimmed(
      self,
  ):
    output_data = experiment_design.trim_outliers(
        data=self.data,
        order_by="a",
        trim_percentile_top=0.001,
        trim_percentile_bottom=0.001,
        rng=self.rng,
    )
    pd.testing.assert_frame_equal(output_data, self.data)

  def test_expected_rows_are_trimmed(self):
    output_data = experiment_design.trim_outliers(
        data=self.data,
        order_by="a",
        trim_percentile_top=0.2,  # Remove top 20% rows (2 rows)
        trim_percentile_bottom=0.1,  # Remove bottom 10% of rows (1 row)
        rng=self.rng,
    )

    expected_data = self.data[
        (self.data["a_rank"] >= 2) & (self.data["a_rank"] <= 8)
    ]
    pd.testing.assert_frame_equal(output_data, expected_data)


if __name__ == "__main__":
  absltest.main()
