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

  def test_plot_metric_history_returns_3_axes(self):
    ax = plotting.plot_metric_history(self.plot_data, "impressions")
    self.assertTupleEqual(ax.shape, (3,))

  def test_plot_metric_history_returns_array_of_axes(self):
    ax = plotting.plot_metric_history(self.plot_data, "impressions")
    self.assertIsInstance(ax[0], plt.Axes)
    self.assertIsInstance(ax[1], plt.Axes)
    self.assertIsInstance(ax[2], plt.Axes)


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
        0.5, aa_simulation_results, ab_simulation_results, 0.05
    )
    self.analysis_results_none = MockSimulationAnalysis(
        0.5, aa_simulation_results, ab_simulation_results, None
    )

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
      plotting.plot_deep_dive(self.analysis_results_none)


class AddCupedAdjustedMetricPerDateTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_data = pd.DataFrame(
        data={
            "item_id": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
            "date": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
            ],
            "time_period": [
                "pretest",
                "test",
                "test",
                "pretest",
                "test",
                "test",
                "pretest",
                "test",
                "test",
            ],
            "metric": [25, 26, 17, 28, 27, 29, 17, 28, 16],
        }
    )

  def test_cuped_adjustment(self):
    result_data = plotting._add_cuped_adjusted_metric_per_date(
        self.mock_data, "metric", "item_id", "date", "time_period"
    )

    expected_data = pd.DataFrame(
        data={
            "item_id": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
            "date": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
            ],
            "time_period": [
                "pretest",
                "test",
                "test",
                "pretest",
                "test",
                "test",
                "pretest",
                "test",
                "test",
            ],
            "metric": [25, 26, 17, 28, 27, 29, 17, 28, 16],
            "metric_cuped_adjusted": [
                23.333333,
                26.206186,
                15.393471,
                23.333333,
                27.57732,
                24.501718,
                23.333333,
                27.216495,
                22.104811,
            ],
        }
    )
    pd.testing.assert_frame_equal(result_data, expected_data)

  def test_raises_exception_if_no_pretest_data(self):
    self.mock_data["time_period"] = "test"

    with self.assertRaises(ValueError):
      plotting._add_cuped_adjusted_metric_per_date(
          self.mock_data, "metric", "item_id", "date", "time_period"
      )


@dataclasses.dataclass
class MockExperimentDesign:
  runtime_weeks: int | None
  pretest_weeks: int | None
  crossover_washout_weeks: int | None
  alpha: float | None
  is_crossover: bool | None


class PlotMetricOverTimeTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    runtime_weeks = 8
    pretest_weeks = 2
    crossover_washout_weeks = 1
    alpha = 0.05
    is_crossover = True

    self.design_crossover = MockExperimentDesign(
        runtime_weeks=runtime_weeks,
        pretest_weeks=pretest_weeks,
        crossover_washout_weeks=crossover_washout_weeks,
        alpha=alpha,
        is_crossover=is_crossover,
    )
    self.design_traditional = MockExperimentDesign(
        runtime_weeks=runtime_weeks,
        pretest_weeks=pretest_weeks,
        crossover_washout_weeks=None,
        alpha=alpha,
        is_crossover=False,
    )

    self.tradtional_data = pd.DataFrame(
        data={
            "metric": [
                25,
                26,
                17,
                28,
                27,
                29,
                17,
                28,
                16,
                34,
                456,
                34,
                54,
                56,
                23,
            ],
            "item_id": [
                "A",
                "A",
                "A",
                "A",
                "A",
                "B",
                "B",
                "B",
                "B",
                "B",
                "C",
                "C",
                "C",
                "C",
                "C",
            ],
            "time_period": [
                "pretest",
                "test",
                "test",
                "test",
                "test",
                "pretest",
                "test",
                "test",
                "test",
                "test",
                "pretest",
                "test",
                "test",
                "test",
                "test",
            ],
            "treatment_assignment": [
                "Control",
                "Control",
                "Control",
                "Control",
                "Control",
                "Treatment",
                "Treatment",
                "Treatment",
                "Treatment",
                "Treatment",
                "Treatment",
                "Treatment",
                "Treatment",
                "Treatment",
                "Treatment",
            ],
            "date": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
            ],
        }
    )
    self.tradtional_data["date"] = pd.to_datetime(self.tradtional_data["date"])

    self.crossover_data = pd.DataFrame(
        data={
            "metric": [
                25,
                26,
                17,
                28,
                27,
                29,
                17,
                28,
                16,
                34,
                456,
                34,
                54,
                56,
                23,
            ],
            "item_id": [
                "A",
                "A",
                "A",
                "A",
                "A",
                "B",
                "B",
                "B",
                "B",
                "B",
                "C",
                "C",
                "C",
                "C",
                "C",
            ],
            "time_period": [
                "pretest",
                "washout_1",
                "test_1",
                "washout_2",
                "test_2",
                "pretest",
                "washout_1",
                "test_1",
                "washout_2",
                "test_2",
                "pretest",
                "washout_1",
                "test_1",
                "washout_2",
                "test_2",
            ],
            "treatment_assignment": [
                "Control",
                "Control",
                "Control",
                "Control",
                "Control",
                "Treatment",
                "Treatment",
                "Treatment",
                "Treatment",
                "Treatment",
                "Control",
                "Control",
                "Control",
                "Control",
                "Control",
            ],
            "date": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
            ],
        }
    )
    self.crossover_data["date"] = pd.to_datetime(self.crossover_data["date"])

  @parameterized.parameters(True, False)
  def test_plot_metric_over_time_for_crossover_design(
      self, apply_cuped_adjustment
  ):
    fig, ax = plotting.plot_metric_over_time(
        data=self.crossover_data,
        metric_column="metric",
        item_id_column="item_id",
        time_period_column="time_period",
        treatment_assignment_column="treatment_assignment",
        date_column="date",
        design=self.design_crossover,
        apply_cuped_adjustment=apply_cuped_adjustment,
    )
    self.assertIsInstance(fig, plt.Figure)
    self.assertIsInstance(ax, np.ndarray)
    self.assertIsInstance(ax[0], plt.Axes)
    self.assertIsInstance(ax[1], plt.Axes)

  @parameterized.parameters(True, False)
  def test_plot_metric_over_time_for_traditional_design(
      self, apply_cuped_adjustment
  ):
    fig, ax = plotting.plot_metric_over_time(
        data=self.tradtional_data,
        metric_column="metric",
        item_id_column="item_id",
        time_period_column="time_period",
        treatment_assignment_column="treatment_assignment",
        date_column="date",
        design=self.design_traditional,
        apply_cuped_adjustment=apply_cuped_adjustment,
    )
    self.assertIsInstance(fig, plt.Figure)
    self.assertIsInstance(ax, np.ndarray)
    self.assertIsInstance(ax[0], plt.Axes)
    self.assertIsInstance(ax[1], plt.Axes)


@dataclasses.dataclass
class MockExperimentDesignPlot:
  primary_metric: str | None


class PlotEffectsTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.mock_results = pd.DataFrame({
        "metric": [
            "Clicks",
            "Impressions",
            "Clicks (no trimm)",
            "CTR",
            "CTR (no trim)",
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
        "is_significant": [True, False, True, False, False],
    }).set_index("metric")

    self.mock_design = MockExperimentDesignPlot(primary_metric="clicks")
    self.mock_design_none = MockExperimentDesignPlot(
        primary_metric="conversions"
    )

  def test_plot_effects_returns_2_axes(self):
    ax = plotting.plot_effects(
        self.mock_results,
        design=self.mock_design,
    )
    self.assertTupleEqual(ax.shape, (2,))

  def test_plot_effects_returns_array_of_axes(self):
    ax = plotting.plot_effects(
        self.mock_results,
        design=self.mock_design,
    )
    self.assertIsInstance(ax[0], plt.Axes)
    self.assertIsInstance(ax[1], plt.Axes)

  def test_raises_exception_if_no_primary_metric(self):
    with self.assertRaises(ValueError):
      plotting.plot_effects(
          self.mock_results,
          design=self.mock_design_none,
      )


if __name__ == "__main__":
  absltest.main()
