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

"""Tests for plotting."""

from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from feedx import plotting


class PlotMetricHistoryTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.plot_data = pd.DataFrame({
        "item_id": [1] * 6 + [2] * 6 + [3] * 6 + [4] * 6 + [5] * 6,
        "week_start": pd.to_datetime(
            [
                "2023-10-02",
                "2023-10-09",
                "2023-10-16",
                "2023-10-23",
                "2023-10-30",
                "2023-11-06",
            ]
            * 5
        ),
        "clicks": (
            [0, 1, 5, 3, 2, 5]
            + [0, 0, 0, 1, 0, 0]
            + [5, 1, 0, 2, 5, 1]
            + [2, 2, 1, 2, 4, 5]
            + [9, 1, 2, 0, 0, 1]
        ),
        "impressions": (
            [0, 1, 5, 3, 2, 5]
            + [0, 0, 0, 1, 0, 0]
            + [5, 1, 0, 2, 5, 1]
            + [2, 2, 1, 2, 4, 5]
            + [9, 1, 2, 0, 0, 1]
        ),
    })

  def test_plot_metric_history_returns_array(self):
    ax = plotting.plot_metric_history(self.plot_data, "impressions")
    self.assertIsInstance(ax, np.ndarray)

  def test_plot_metric_history_returns_2_axes(self):
    ax = plotting.plot_metric_history(self.plot_data, "impressions")
    self.assertTupleEqual(ax.shape, (2,))

  def test_plot_metric_history_returns_array_of_axes(self):
    ax = plotting.plot_metric_history(self.plot_data, "impressions")
    self.assertIsInstance(ax[0], plt.Axes)
    self.assertIsInstance(ax[1], plt.Axes)


if __name__ == "__main__":
  absltest.main()
