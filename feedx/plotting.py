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

"""Plot historical and simulation data."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from feedx import experiment_simulations


def _make_density_plot(
    data: pd.DataFrame,
    column: str,
    ax: plt.Axes,
    x_label: str | None = None,
    y_label: str | None = None,
    color: str = "C0",
    label: str | None = None,
    edgecolor: str | None = None,
) -> None:
  """Make a density plot using matplotlib visualizing distribution of the metric.

  Args:
    data: data for plotting
    column: name of metric column in the data (eg clicks or impressions)
    ax: class containing the plotted data of the data set provided
    x_label: x-axis label. If not provided, column will be displayed
    y_label: y-axis label. If not provided, "Density" will be displayed
    color: color in which to display the data in the plots
    label: optional value to override x_label in the plot legend
    edgecolor: color in which to display the edges of the density plots
  """
  x_label = x_label or column
  y_label = y_label or "Number of Items"

  ax.hist(
      data[column],
      bins="auto",
      histtype="bar",
      alpha=0.5,
      color=color,
      label=label,
      edgecolor=edgecolor,
  )
  ax.set_title(f"Density Plot of {x_label} Distribution")
  ax.set_xlabel(f"{x_label}")
  ax.set_ylabel(f"{y_label}")


def plot_metric_history(
    data: pd.DataFrame,
    metric_column: str,
    week_start_column_name: str = "week_start",
) -> np.ndarray:
  """Plots the weekly distribution and the average of the metric per item per week.

  Args:
    data: data set to plot containing the week_start column and the metrics that
      will be evaluated
    metric_column: metric chosen to run the plot
    week_start_column_name: column containing the week start date

  Returns:
    The axes containing the plots of the metric performance history
  """

  _, axs = plt.subplots(
      nrows=1, ncols=2, figsize=(15, 5), constrained_layout=True
  )

  clean_metric_name = metric_column.replace("_", " ")

  metric_time = (
      data.groupby(week_start_column_name)[[metric_column]].sum().reset_index()
  )
  axs[0].plot(metric_time[week_start_column_name], metric_time[metric_column])
  axs[0].set_title(f"Total {clean_metric_name} per week")
  axs[0].set(xlabel="Start Week", ylabel=f"Total {clean_metric_name}")
  axs[0].tick_params(axis="x", rotation=30)

  _make_density_plot(
      data,
      ax=axs[1],
      column=metric_column,
      x_label=f"Average {clean_metric_name} per item per week",
  )

  return axs


def plot_deep_dive(
    design_results: experiment_simulations.SimulationAnalysis,
) -> np.ndarray:
  """Plots the effects density and p-values distribution of the A/A and A/B simulation results.

  Args:
    design_results: the simulation analysis results generated from the selected
      experiment design

  Returns:
    The effects density plot of the A/A and A/B simulation results
    The distribution plot of the p-values from the A/A results which should be
      uniformially distributed between 0 and 1
    The distribution plot of the p-values from the A/B results which should be
      skewed towards zero

  Raises:
    ValueError: validation for selected designs has not been run. The
      experiments must be simulated prior to running this function.
  """
  if design_results.simulated_false_positive_rate is None:
    raise ValueError(
        "Cannot plot deep dive because the validation has not been run."
    )
  _, ax = plt.subplots(
      nrows=1, ncols=3, figsize=(15, 5), constrained_layout=True
  )

  # Effects density plots

  # A/A effect distribution
  zero_line = 0.0
  _make_density_plot(
      design_results.aa_simulation_results,
      column="absolute_difference",
      x_label="Effect Size",
      y_label="Density",
      color="C0",
      ax=ax[0],
      label="A/A absolute difference",
  )
  ax[0].axvline(x=zero_line, color="C0", linestyle="--", linewidth=2)

  # A/B effect distribution
  mde = design_results.minimum_detectable_effect
  _make_density_plot(
      design_results.ab_simulation_results,
      column="absolute_difference",
      x_label="Effect Size",
      y_label="Density",
      color="C1",
      ax=ax[0],
      label="A/B absolute difference",
  )
  ax[0].axvline(x=mde, color="C1", linestyle="--", linewidth=2)

  ax[0].legend()
  ax[0].set_title(
      "\n".join([
          "Check that effect estimates (shaded areas)",
          "are centered on the true effects (dashed lines)",
      ])
  )

  # A/A p-value distribution
  _make_density_plot(
      design_results.aa_simulation_results,
      column="p_value",
      x_label="P-Values (A/A)",
      y_label="Count",
      color="C0",
      ax=ax[1],
      edgecolor="black",
  )
  ax[1].set_title(
      "\n".join([
          "Check that the p-values from the A/A test",
          "are uniformly distributed between 0 and 1.",
      ])
  )

  # A/B p-value distribution
  _make_density_plot(
      design_results.ab_simulation_results,
      column="p_value",
      x_label="P-Values (A/B)",
      y_label="Count",
      color="C1",
      ax=ax[2],
      edgecolor="black",
  )
  ax[2].set_title(
      "\n".join(
          ["Check that the p-values from the A/B test", "are skewed towards 0."]
      )
  )

  return ax
