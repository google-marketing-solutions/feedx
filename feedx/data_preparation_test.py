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
import numpy as np
import pandas as pd

from feedx import data_preparation


class SyntheticDataTest(parameterized.TestCase):

  def test_generate_synthetic_data_generates_expected_number_of_rows(
      self,
  ):
    n_items = 500
    weeks_before_start_date = 4
    weeks_after_start_date = 8
    n_days = (weeks_before_start_date + weeks_after_start_date) * 7

    rng = np.random.default_rng(123)

    data = data_preparation.generate_synthetic_data(
        rng=rng,
        n_items=n_items,
        weeks_before_start_date=weeks_before_start_date,
        weeks_after_start_date=weeks_after_start_date,
    )

    self.assertLen(data, n_items * n_days)

  def test_generate_synthetic_data_has_expected_columns(self):
    data = data_preparation.generate_synthetic_data(
        rng=np.random.default_rng(123)
    )

    self.assertCountEqual(
        data.columns.values,
        [
            "item_id",
            "date",
            "clicks",
            "impressions",
            "total_cost",
            "conversions",
            "total_conversion_value",
        ],
    )

  @parameterized.parameters([-0.1, 0.0])
  def test_generate_synthetic_data_raises_exception_for_bad_impressions_average(
      self, impressions_average
  ):
    with self.assertRaises(ValueError):
      data_preparation.generate_synthetic_data(
          rng=np.random.default_rng(123),
          impressions_average=impressions_average,
      )

  @parameterized.parameters([-0.1, 0.0])
  def test_generate_synthetic_data_raises_exception_for_bad_impressions_standard_deviation(
      self, impressions_standard_deviation
  ):
    with self.assertRaises(ValueError):
      data_preparation.generate_synthetic_data(
          rng=np.random.default_rng(123),
          impressions_standard_deviation=impressions_standard_deviation,
      )

  @parameterized.parameters([-0.1, 0.0, 1.0, 1.1])
  def test_generate_synthetic_data_raises_exception_for_bad_ctr_average(
      self, ctr_average
  ):
    with self.assertRaises(ValueError):
      data_preparation.generate_synthetic_data(
          rng=np.random.default_rng(123),
          ctr_average=ctr_average,
      )

  @parameterized.parameters([-0.1, 0.0, 1.0, 1.1])
  def test_generate_synthetic_data_raises_exception_for_bad_ctr_standard_deviation(
      self, ctr_standard_deviation
  ):
    with self.assertRaises(ValueError):
      data_preparation.generate_synthetic_data(
          rng=np.random.default_rng(123),
          ctr_standard_deviation=ctr_standard_deviation,
      )


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


class DownsampleItemsTests(parameterized.TestCase):

  def setUp(self):
    self.data = pd.DataFrame({
        "item_id": ["1", "1", "2", "2", "3", "3", "4", "4"],
        "clicks": [1, 2, 3, 4, 5, 6, 7, 8],
    })
    self.rng = np.random.default_rng(123)

  def test_downsample_items_downsamples_if_downsample_fraction_less_than_1(
      self,
  ):
    downsampled_data = data_preparation.downsample_items(
        self.data,
        downsample_fraction=0.5,
        item_id_column="item_id",
        rng=self.rng,
    )
    self.assertLess(
        len(downsampled_data.index.values), len(self.data.index.values)
    )

  def test_downsample_items_keeps_all_original_rows_for_sampled_items(
      self,
  ):
    downsampled_data = data_preparation.downsample_items(
        self.data,
        downsample_fraction=0.5,
        item_id_column="item_id",
        rng=self.rng,
    )

    original_data_for_sampled_items = self.data[
        self.data["item_id"].isin(downsampled_data["item_id"])
    ]

    pd.testing.assert_frame_equal(
        original_data_for_sampled_items, downsampled_data, check_like=True
    )

  def test_downsample_items_does_not_downsample_if_downsample_fraction_is_1(
      self,
  ):
    downsampled_data = data_preparation.downsample_items(
        self.data,
        downsample_fraction=1.0,
        item_id_column="item_id",
        rng=self.rng,
    )
    pd.testing.assert_frame_equal(self.data, downsampled_data, check_like=True)

  def test_downsample_items_raises_exception_if_item_id_column_is_missing(
      self,
  ):
    self.data.drop("item_id", axis=1, inplace=True)
    with self.assertRaises(ValueError):
      data_preparation.downsample_items(
          self.data,
          downsample_fraction=0.5,
          item_id_column="item_id",
          rng=self.rng,
      )

  @parameterized.parameters(-0.1, 0.0, 1.1)
  def test_downsample_items_raises_exception_for_bad_downsample_fraction(
      self, bad_downsample_fraction
  ):
    with self.assertRaises(ValueError):
      data_preparation.downsample_items(
          self.data,
          downsample_fraction=bad_downsample_fraction,
          item_id_column="item_id",
          rng=self.rng,
      )


class DataValidationTests(parameterized.TestCase):

  def test_validate_historical_data_passes_for_good_daily_data(self):
    historical_data = pd.DataFrame({
        "date": pd.to_datetime(
            ["2023-10-01", "2023-10-01", "2023-10-02", "2023-10-02"]
        ),
        "date_id": [1, 1, 2, 2],
        "item_id": ["1", "2", "1", "2"],
        "clicks": [1, 2, 3, 4],
    })

    data_preparation.validate_historical_data(
        historical_data,
        item_id_column="item_id",
        date_column="date",
        date_id_column="date_id",
        primary_metric_column="clicks",
    )

  def test_validate_historical_data_passes_for_good_weekly_data(self):
    historical_data = pd.DataFrame({
        "date": pd.to_datetime(
            ["2023-10-01", "2023-10-01", "2023-10-08", "2023-10-08"]
        ),
        "date_id": [1, 1, 2, 2],
        "item_id": ["1", "2", "1", "2"],
        "clicks": [1, 2, 3, 4],
    })

    data_preparation.validate_historical_data(
        historical_data,
        item_id_column="item_id",
        date_column="date",
        date_id_column="date_id",
        primary_metric_column="clicks",
    )

  def test_historical_data_must_have_unique_item_id_and_dates(self):
    bad_historical_data = pd.DataFrame({
        "date": pd.to_datetime([
            "2023-10-01",
            "2023-10-01",
            "2023-10-02",
            "2023-10-02",
            "2023-10-02",
        ]),
        "date_id": [1, 1, 2, 2, 2],
        "item_id": ["1", "2", "1", "2", "2"],
        "clicks": [1, 2, 3, 4, 5],
    })

    with self.assertRaises(ValueError):
      data_preparation.validate_historical_data(
          bad_historical_data,
          item_id_column="item_id",
          date_column="date",
          date_id_column="date_id",
          primary_metric_column="clicks",
      )

  def test_historical_data_must_have_every_item_and_date_combination(self):
    bad_historical_data = pd.DataFrame({
        "date": pd.to_datetime(["2023-10-01", "2023-10-01", "2023-10-02"]),
        "date_id": [1, 1, 2],
        "item_id": ["1", "2", "1"],
        "clicks": [1, 2, 3],
    })

    with self.assertRaises(ValueError):
      data_preparation.validate_historical_data(
          bad_historical_data,
          item_id_column="item_id",
          date_column="date",
          date_id_column="date_id",
          primary_metric_column="clicks",
      )

  def test_historical_data_must_have_no_nulls(self):
    bad_historical_data = pd.DataFrame({
        "date": pd.to_datetime(
            ["2023-10-01", "2023-10-01", "2023-10-02", "2023-10-02"]
        ),
        "date_id": [1, 1, 2, 2],
        "item_id": ["1", "2", "1", "2"],
        "clicks": [1, 2, 3, None],
    })

    with self.assertRaises(ValueError):
      data_preparation.validate_historical_data(
          bad_historical_data,
          item_id_column="item_id",
          date_column="date",
          date_id_column="date_id",
          primary_metric_column="clicks",
      )

  def test_historical_data_must_be_either_daily_or_weekly_spaced(self):
    bad_historical_data = pd.DataFrame({
        "date": pd.to_datetime(
            ["2023-10-01", "2023-10-01", "2023-10-03", "2023-10-03"]
        ),
        "date_id": [1, 1, 2, 2],
        "item_id": ["1", "2", "1", "2"],
        "clicks": [1, 2, 3, 4],
    })

    with self.assertRaises(ValueError):
      data_preparation.validate_historical_data(
          bad_historical_data,
          item_id_column="item_id",
          date_column="date",
          date_id_column="date_id",
          primary_metric_column="clicks",
      )

  def test_historical_data_date_id_must_be_aligned_with_date(self):
    bad_historical_data = pd.DataFrame({
        "date": pd.to_datetime(
            ["2023-10-01", "2023-10-01", "2023-10-03", "2023-10-03"]
        ),
        "date_id": [2, 2, 1, 1],
        "item_id": ["1", "2", "1", "2"],
        "clicks": [1, 2, 3, 4],
    })

    with self.assertRaises(ValueError):
      data_preparation.validate_historical_data(
          bad_historical_data,
          item_id_column="item_id",
          date_column="date",
          date_id_column="date_id",
          primary_metric_column="clicks",
      )

  def test_historical_data_primary_metric_must_be_finite(self):
    bad_historical_data = pd.DataFrame({
        "date": pd.to_datetime(
            ["2023-10-01", "2023-10-01", "2023-10-02", "2023-10-02"]
        ),
        "date_id": [1, 1, 2, 2],
        "item_id": ["1", "2", "1", "2"],
        "clicks": [1, 2, 3, np.inf],
    })

    with self.assertRaises(ValueError):
      data_preparation.validate_historical_data(
          bad_historical_data,
          item_id_column="item_id",
          date_column="date",
          date_id_column="date_id",
          primary_metric_column="clicks",
      )

  def test_historical_data_metrics_must_be_non_negative_by_default(self):
    bad_historical_data = pd.DataFrame({
        "date": pd.to_datetime(
            ["2023-10-01", "2023-10-01", "2023-10-02", "2023-10-02"]
        ),
        "date_id": [1, 1, 2, 2],
        "item_id": ["1", "2", "1", "2"],
        "clicks": [1, 2, 3, -1],
    })

    with self.assertRaises(ValueError):
      data_preparation.validate_historical_data(
          bad_historical_data,
          item_id_column="item_id",
          date_column="date",
          date_id_column="date_id",
          primary_metric_column="clicks",
      )

  def test_historical_data_metrics_may_be_non_negative_if_require_positive_primary_metric_is_false(
      self,
  ):
    good_historical_data = pd.DataFrame({
        "date": pd.to_datetime(
            ["2023-10-01", "2023-10-01", "2023-10-02", "2023-10-02"]
        ),
        "date_id": [1, 1, 2, 2],
        "item_id": ["1", "2", "1", "2"],
        "clicks": [1, 2, 3, -1],
    })

    data_preparation.validate_historical_data(
        good_historical_data,
        item_id_column="item_id",
        date_column="date",
        date_id_column="date_id",
        primary_metric_column="clicks",
        require_positive_primary_metric=False,
    )

  def test_historical_data_date_id_must_be_consecutive(
      self,
  ):
    good_historical_data = pd.DataFrame({
        "date": pd.to_datetime(
            ["2023-10-01", "2023-10-01", "2023-10-02", "2023-10-02"]
        ),
        "date_id": [1, 1, 3, 3],
        "item_id": ["1", "2", "1", "2"],
        "clicks": [1, 2, 3, -1],
    })
    with self.assertRaises(ValueError):
      data_preparation.validate_historical_data(
          good_historical_data,
          item_id_column="item_id",
          date_column="date",
          date_id_column="date_id",
          primary_metric_column="clicks",
          require_positive_primary_metric=False,
      )

  def test_historical_data_date_id_must_be_an_integer(
      self,
  ):
    good_historical_data = pd.DataFrame({
        "date": pd.to_datetime(
            ["2023-10-01", "2023-10-01", "2023-10-02", "2023-10-02"]
        ),
        "date_id": ["1", "1", "2", "2"],
        "item_id": ["1", "2", "1", "2"],
        "clicks": [1, 2, 3, -1],
    })

    with self.assertRaises(ValueError):
      data_preparation.validate_historical_data(
          good_historical_data,
          item_id_column="item_id",
          date_column="date",
          date_id_column="date_id",
          primary_metric_column="clicks",
          require_positive_primary_metric=False,
      )


class AddWeekIdAndWeekStartTests(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.data = pd.DataFrame({
        "date": pd.to_datetime([
            "2023-10-01",
            "2023-10-02",
            "2023-10-07",
            "2023-10-08",
            "2023-10-09",
            "2023-10-17",
        ]),
    })

  def test_adds_week_id_relative_to_first_date_by_default(self):
    actual_result = data_preparation.add_week_id_and_week_start(
        self.data, date_column="date"
    )

    expected_result = pd.DataFrame({
        "date": pd.to_datetime([
            "2023-10-01",
            "2023-10-02",
            "2023-10-07",
            "2023-10-08",
            "2023-10-09",
            "2023-10-17",
        ]),
        "week_id": [0, 0, 0, 1, 1, 2],
        "week_start": pd.to_datetime([
            "2023-10-01",
            "2023-10-01",
            "2023-10-01",
            "2023-10-08",
            "2023-10-08",
            "2023-10-15",
        ]),
    })

    pd.testing.assert_frame_equal(
        actual_result, expected_result, check_like=True
    )

  def test_adds_week_id_relative_to_experiment_start_date_if_set(self):
    actual_result = data_preparation.add_week_id_and_week_start(
        self.data, date_column="date", experiment_start_date="2023-10-02"
    )

    expected_result = pd.DataFrame({
        "date": pd.to_datetime([
            "2023-10-01",
            "2023-10-02",
            "2023-10-07",
            "2023-10-08",
            "2023-10-09",
            "2023-10-17",
        ]),
        "week_id": [-1, 0, 0, 0, 1, 2],
        "week_start": pd.to_datetime([
            "2023-09-25",
            "2023-10-02",
            "2023-10-02",
            "2023-10-02",
            "2023-10-09",
            "2023-10-16",
        ]),
    })

    pd.testing.assert_frame_equal(
        actual_result, expected_result, check_like=True
    )


class GroupDataToCompleteWeeksTests(parameterized.TestCase):
  def test_group_daily_data_to_weekly(self):
    values = [
        ["ABC123", "2023-10-01", 1, 10000],
        ["ABC123", "2023-10-02", 50, 500],
        ["ABC123", "2023-10-03", 100, 3245234],
        ["ABC123", "2023-10-04", 435, 353245],
        ["ABC123", "2023-10-05", 123, 3245],
        ["ABC123", "2023-10-06", 35, 2],
        ["ABC123", "2023-10-07", 25, 325],
        ["ABC123", "2023-10-08", 34, 354],
    ]
    daily_data = pd.DataFrame(
        values, columns=["item_id", "date", "clicks", "impressions"]
    )
    daily_data["date"] = pd.to_datetime(daily_data["date"])
    daily_data = data_preparation.add_week_id_and_week_start(
        daily_data, date_column="date"
    )

    expected_daily_result = pd.DataFrame([{
        "week_id": 0,
        "week_start": "2023-10-01",
        "item_id": "ABC123",
        "clicks": 769,
        "impressions": 3612551,
    }])
    expected_daily_result["week_start"] = pd.to_datetime(
        expected_daily_result["week_start"]
    )
    actual_daily_result = data_preparation.group_data_to_complete_weeks(
        daily_data,
        item_id_column="item_id",
        date_column="date",
        week_id_column="week_id",
        week_start_column="week_start",
    )
    pd.testing.assert_frame_equal(
        actual_daily_result, expected_daily_result, check_like=True
    )

  def test_group_weekly_data_to_weekly(self):
    values = [
        ["ABC123", "2023-10-01", 1, 10000],
        ["ABC123", "2023-10-08", 50, 500],
        ["ABC123", "2023-10-15", 100, 3245234],
        ["ABC123", "2023-10-22", 435, 353245],
    ]
    weekly_data = pd.DataFrame(
        values, columns=["item_id", "date", "clicks", "impressions"]
    )
    weekly_data["date"] = pd.to_datetime(weekly_data["date"])
    weekly_data = data_preparation.add_week_id_and_week_start(
        weekly_data, date_column="date"
    )

    expected_weekly_result = pd.DataFrame([
        {
            "item_id": "ABC123",
            "week_start": "2023-10-01",
            "clicks": 1,
            "impressions": 10000,
            "week_id": 0,
        },
        {
            "item_id": "ABC123",
            "week_start": "2023-10-08",
            "clicks": 50,
            "impressions": 500,
            "week_id": 1,
        },
        {
            "item_id": "ABC123",
            "week_start": "2023-10-15",
            "clicks": 100,
            "impressions": 3245234,
            "week_id": 2,
        },
        {
            "item_id": "ABC123",
            "week_start": "2023-10-22",
            "clicks": 435,
            "impressions": 353245,
            "week_id": 3,
        },
    ])
    expected_weekly_result["week_start"] = pd.to_datetime(
        expected_weekly_result["week_start"])

    actual_weekly_result = data_preparation.group_data_to_complete_weeks(
        weekly_data,
        item_id_column="item_id",
        date_column="date",
        week_id_column="week_id",
        week_start_column="week_start",
    )
    pd.testing.assert_frame_equal(
        actual_weekly_result, expected_weekly_result, check_like=True
    )

  def test_group_mixed_data_to_weekly_raises_exception_if_mixed_frequency(self):
    values = [
        ["ABC123", "2023-10-01", 1, 10000],
        ["ABC123", "2023-10-08", 50, 500],
        ["ABC123", "2023-10-15", 100, 3245234],
        ["ABC123", "2023-10-22", 435, 353245],
        ["ABC123", "2023-10-02", 50, 500],
        ["ABC123", "2023-10-03", 100, 3245234],
        ["ABC123", "2023-10-04", 435, 353245],
        ["ABC123", "2023-10-05", 123, 3245],
        ["ABC123", "2023-10-06", 35, 2],
        ["ABC123", "2023-10-07", 25, 325],
    ]
    mixed_data = pd.DataFrame(
        values, columns=["item_id", "date", "clicks", "impressions"]
    )
    mixed_data["date"] = pd.to_datetime(mixed_data["date"])
    mixed_data = data_preparation.add_week_id_and_week_start(
        mixed_data, date_column="date"
    )

    with self.assertRaisesRegex(
        ValueError,
        "The data is not daily or weekly. Cannot proceed with mixed frequency",
    ):
      data_preparation.group_data_to_complete_weeks(
          mixed_data,
          item_id_column="item_id",
          date_column="date",
          week_id_column="week_id",
          week_start_column="week_start",
      )


if __name__ == "__main__":
  absltest.main()
