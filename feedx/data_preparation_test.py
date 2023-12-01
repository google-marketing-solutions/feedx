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

"""Tests for data_preparation."""

from absl.testing import absltest
from absl.testing import parameterized
import pandas as pd

from feedx import data_preparation


class SyntheticDataTest(parameterized.TestCase):

  def test_generate_historical_synthetic_data(self):
    # TODO(sam-bailey): Improve synthetic data tests.
    number_samples = 500
    hist_days = 90
    data = data_preparation.generate_historical_synthetic_data(
        n_items=number_samples, historical_days=hist_days
    )
    self.assertEqual(len(data), number_samples * hist_days)


class StandardizeColumnNamesAndTypesTests(parameterized.TestCase):

  def setUp(self):
    super().setUp()

    self.data = pd.DataFrame({
        "Item ID": [1, 2, 3],
        "YYYY-MM-DD": ["2023-10-01", "2023-10-02", "2023-10-03"],
        "Clicks": ["1", "2", "3"],
        "Impr.": [20.0, 21.0, 22.0],
        "Conv.": ["2", "3", "5"],
        "Conv. Value": ["2.3", "4.2", "5.0"],
        "Cost": [1, 2, 3],
        "Some extra column": ["something", "something", "something"],
    })

  def test_standardize_column_names_and_types_standardizes_columns_as_expected(
      self,
  ):
    parsed_data = data_preparation.standardize_column_names_and_types(
        self.data,
        item_id_column="Item ID",
        date_column="YYYY-MM-DD",
        clicks_column="Clicks",
        impressions_column="Impr.",
        conversions_column="Conv.",
        total_conversion_value_column="Conv. Value",
        total_cost_column="Cost",
    )
    expected_data = pd.DataFrame({
        "item_id": ["1", "2", "3"],
        "date": [
            pd.to_datetime("2023-10-01"),
            pd.to_datetime("2023-10-02"),
            pd.to_datetime("2023-10-03"),
        ],
        "clicks": [1, 2, 3],
        "impressions": [20, 21, 22],
        "conversions": [2, 3, 5],
        "total_conversion_value": [2.3, 4.2, 5.0],
        "total_cost": [1.0, 2.0, 3.0],
    })
    pd.testing.assert_frame_equal(parsed_data, expected_data, check_like=True)


class FillMissingRowsWithZerosTests(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    # Input data is missing item 2 on 2023-10-02 and item 3 on 2023-10-01
    self.input_data = pd.DataFrame({
        "item_id": ["1", "1", "2", "3"],
        "date": ["2023-10-01", "2023-10-02", "2023-10-01", "2023-10-02"],
        "clicks": [1, 2, 3, 4],
        "total_conversion_value": [20.0, 21.0, 22.0, 22.0],
    })

  def test_fill_missing_rows_with_zeros_adds_expected_missing_rows(self):
    actual_result = data_preparation.fill_missing_rows_with_zeros(
        self.input_data, item_id_column="item_id", date_column="date"
    )

    expected_result = pd.DataFrame({
        "item_id": ["1", "1", "2", "2", "3", "3"],
        "date": [
            "2023-10-01",
            "2023-10-02",
            "2023-10-01",
            "2023-10-02",
            "2023-10-01",
            "2023-10-02",
        ],
        "clicks": [1, 2, 3, 0, 0, 4],
        "total_conversion_value": [20.0, 21.0, 22.0, 0.0, 0.0, 22.0],
    })
    pd.testing.assert_frame_equal(
        actual_result, expected_result, check_like=True
    )

  @parameterized.parameters(["item_id", "date"])
  def test_fill_missing_rows_with_zeros_raises_exception_if_required_column_is_missing(
      self, required_column_name
  ):
    self.input_data.drop(required_column_name, axis=1, inplace=True)
    with self.assertRaises(ValueError):
      data_preparation.fill_missing_rows_with_zeros(
          self.input_data, item_id_column="item_id", date_column="date"
      )


if __name__ == "__main__":
  absltest.main()
