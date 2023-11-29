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

import datetime as dt
import io
import unittest

import pandas as pd
from feedx import data_preparation


class SyntheticDataTest(unittest.TestCase):

  def test_generate_historical_synthetic_data(self):
    number_samples = 500
    hist_days = 90
    data = data_preparation.generate_historical_synthetic_data(
        n_items=number_samples, historical_days=hist_days
    )
    self.assertEqual(len(data), number_samples * hist_days)


class CsvDataTest(unittest.TestCase):
  def setUp(self):
    super().setUp()
    self.csv_data = io.StringIO("""Shopping - Item ID
                                "June 26, 2023 - October 1, 2023"
                                Item ID,Week,Clicks,Impr.
                                xxx,2023-08-07,0,0
                                yyy,2023-08-21
                                zzz,2023-08-14,0,0""")


class StandardizeHistoricalDataTest(unittest.TestCase):

  def test_standardize_historical_column_names(self):
    values = [
        ["ABC123", "2023-10-02", 2314, 324523],
        ["XYZ987", "2023-10-01", 1, 10000],
    ]
    data_frame = pd.DataFrame(
        values, columns=["item_id", "date", "clicks", "impressions"]
    )
    data = data_preparation.standardize_historical_column_names(
        data_frame,
        date_column_name=data_frame.columns[0],
        item_id_column_name=data_frame.columns[1],
        clicks_column_name=data_frame.columns[2],
        impressions_column_name=data_frame.columns[3],
    )
    self.assertIsInstance(data, pd.DataFrame)
    self.assertCountEqual(
        data.columns.values.tolist(),
        ["item_id", "date", "clicks", "impressions"],
    )


class ParseDataTest(unittest.TestCase):

  def test_historical_parse_data(self):
    data_frame = pd.DataFrame([
        {
            "SKUs": "ABC123",
            "dates": "2023-10-02",
            "product_clicks": "2314",
            "product_impressions": "324523",
        },
        {
            "SKUs": "ABC123",
            "dates": "2023-10-01",
            "product_clicks": "0",
            "product_impressions": "0",
        },
        {
            "SKUs": "XYZ987",
            "dates": "2023-10-02",
            "product_clicks": "0",
            "product_impressions": "0",
        },
        {
            "SKUs": "XYZ987",
            "dates": "2023-10-01",
            "product_clicks": "1",
            "product_impressions": "10000",
        },
    ])

    columns_dict = {
        "SKUs": "item_id",
        "dates": "date",
        "product_clicks": "clicks",
        "product_impressions": "impressions"
    }

    expected_result = pd.DataFrame([
        {
            "item_id": "ABC123",
            "date": dt.datetime(2023, 10, 2, 0, 0, 0),
            "clicks": 2314,
            "impressions": 324523,
        },
        {
            "item_id": "ABC123",
            "date": dt.datetime(2023, 10, 1, 0, 0, 0),
            "clicks": 0,
            "impressions": 0,
        },
        {
            "item_id": "XYZ987",
            "date": dt.datetime(2023, 10, 2, 0, 0, 0),
            "clicks": 0,
            "impressions": 0,
        },
        {
            "item_id": "XYZ987",
            "date": dt.datetime(2023, 10, 1, 0, 0, 0),
            "clicks": 1,
            "impressions": 10000,
        },
    ])
    actual_result = data_preparation.parse_data(data_frame, columns_dict)

    pd.testing.assert_frame_equal(actual_result, expected_result)


if __name__ == "__main__":
  unittest.main()
