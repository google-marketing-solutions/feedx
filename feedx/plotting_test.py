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

import dataclasses

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


@dataclasses.dataclass
class MockSimulationAnalysis:
  minimum_detectable_effect: float | None
  aa_simulation_results: pd.DataFrame | None
  ab_simulation_results: pd.DataFrame | None
  simulated_false_positive_rate: float | None


class PlotDeepDiveTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    aa_simulation_results = pd.DataFrame({
        "p_value": np.random.rand(1000),
        "absolute_difference": np.random.randn(1000),
    })
    ab_simulation_results = pd.DataFrame({
        "p_value": np.random.rand(1000),
        "absolute_difference": np.random.randn(1000) + 0.5,
    })
    self.analysis_results = MockSimulationAnalysis(
        0.5, aa_simulation_results, ab_simulation_results, 0.05)
    self.analysis_results_none = MockSimulationAnalysis(
        0.5, aa_simulation_results, ab_simulation_results, None)

  def test_plot_deep_dive_returns_array(self):
    ax = plotting.plot_deep_dive(self.analysis_results)
    self.assertIsInstance(ax, np.ndarray)

  def test_plot_deep_dive_returns_3_axes(self):
    ax = plotting.plot_deep_dive(self.analysis_results)
    self.assertTupleEqual(ax.shape, (3,))

  def test_plot_deep_dive_returns_array_of_axes(self):
    ax = plotting.plot_deep_dive(self.analysis_results)
    self.assertIsInstance(ax[0], plt.Axes)
    self.assertIsInstance(ax[1], plt.Axes)
    self.assertIsInstance(ax[2], plt.Axes)

  def test_raises_exception_if_simulation_not_run(self):
    with self.assertRaises(ValueError):
      plotting.plot_deep_dive(
          self.analysis_results_none
      )


if __name__ == "__main__":
  absltest.main()
