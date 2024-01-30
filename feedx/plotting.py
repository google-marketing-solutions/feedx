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

import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from feedx import experiment_design
from feedx import experiment_simulations
from feedx import statistics


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


def _add_cuped_adjusted_metric_per_date(
    data: pd.DataFrame,
    metric_column: str,
    item_id_column: str,
    date_column: str,
    time_period_column: str,
    new_metric_column_name: str | None = None,
) -> pd.DataFrame:
  """Calculates the CUPED-adjusted values of the given metric per item.

  Args:
    data: the input data to analyze
    metric_column: name of the column with the metric to adjust
    item_id_column: name of the column with the item_ids
    date_column: name of the column containing dates
    time_period_column: name of the column containing the test phases, eg
      "pretest", "test_1", "washout_1"
    new_metric_column_name: optional; name of the column of the adjusted metric.
      If it is None, then it will default to the name of the metric column with
      "_cuped_adjusted" appended to it

  Returns:
    Modified data frame with CUPED adjusted metric
  """
  new_metric_column_name = (
      new_metric_column_name or f"{metric_column}_cuped_adjusted"
  )
  if "pretest" not in data[time_period_column].values:
    raise ValueError(
        "CUPED adjustment can only be done with pretest data. Please make sure"
        " to provide pretest data"
    )

  pretest_mask = data[time_period_column] == "pretest"
  pretest_values = (
      data.loc[pretest_mask].groupby(item_id_column)[metric_column].sum()
  )
  covariate = data[item_id_column].map(pretest_values)
  raw_metric = data[metric_column].copy()

  data[new_metric_column_name] = np.empty(len(data), dtype=np.float64)
  for date in data[date_column].drop_duplicates().values:
    date_mask = data[date_column] == date
    data.loc[date_mask, new_metric_column_name] = (
        statistics.apply_cuped_adjustment(
            raw_metric.loc[date_mask].values,
            covariate.loc[date_mask].values,
            0.0,
        )
    )

  return data


def plot_metric_over_time(
    data: pd.DataFrame,
    *,
    metric_column: str,
    item_id_column: str,
    time_period_column: str,
    treatment_assignment_column: str,
    date_column: str,
    design: experiment_design.ExperimentDesign,
    apply_cuped_adjustment: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
  """Plots the peformance of control and treatment groups over time for each metric.

  Args:
    data: the input data to plot
    metric_column: name of the column with the metric to plot
    item_id_column: name of the column with the item_ids
    time_period_column: name of the column containing the test phases, eg
      "pretest", "test_1", "washout_1"
    treatment_assignment_column:
    date_column: name of the column containing dates
    design: experiment design selected for the experiment
    apply_cuped_adjustment: if the function should apply CUPED adjustment before
      plotting

  Returns:
    Plot showing the control and treatment performance over time and and the
      difference between control and treatment for the selected metric.
  """
  data = data.sort_values(date_column)

  if apply_cuped_adjustment:
    data = _add_cuped_adjusted_metric_per_date(
        data.copy(),
        metric_column=metric_column,
        item_id_column=item_id_column,
        date_column=date_column,
        time_period_column=time_period_column,
        new_metric_column_name=metric_column,
    )

  values = pd.pivot_table(
      data=data,
      index=date_column,
      columns=treatment_assignment_column,
      values=metric_column,
      aggfunc="mean",
  ).rename({1: "Treatment", 0: "Control"}, axis=1)

  values_std = pd.pivot_table(
      data=data,
      index=date_column,
      columns=treatment_assignment_column,
      values=metric_column,
      aggfunc=lambda x: np.std(x) / np.sqrt(len(x)),
  ).rename({1: "Treatment", 0: "Control"}, axis=1)

  delta = values["Treatment"] - values["Control"]
  delta_std = np.sqrt(values_std["Treatment"] ** 2 + values_std["Control"] ** 2)
  z = stats.norm.ppf(1.0 - design.alpha / 2)
  delta_lb = delta - z * delta_std
  delta_ub = delta + z * delta_std

  fig, ax = plt.subplots(
      nrows=2, figsize=(10, 6), sharex=True, constrained_layout=True
  )

  is_test_or_test_1 = data["time_period"].isin(["test", "test_1"])
  start_date = data.loc[is_test_or_test_1, "date"].min()
  pretest_start_date = start_date - dt.timedelta(weeks=design.pretest_weeks)
  end_date = start_date + dt.timedelta(weeks=design.runtime_weeks)

  values.plot(lw=1, ax=ax[0], marker=".", label=metric_column.replace("_", " "))
  title = f"Average {metric_column.replace('_', ' ')} per item"
  if apply_cuped_adjustment:
    title += " (cuped adjusted)"
  ax[0].set_title(title)
  ax[0].set_xlim(pretest_start_date, end_date)
  ax[0].set_ylabel(metric_column.replace("_", " ").title())

  delta.plot(lw=1, ax=ax[1], color="C2", marker=".", label="Treatment - Control")
  ax[1].fill_between(
      delta.index.values,
      delta_lb,
      delta_ub,
      color="C2",
      alpha=0.3,
      label=f"{1.0 - design.alpha:.0%} Confidence Interval",
    )
  ax[1].axhline(0.0, color="k", lw=1, ls="--")
  ax[1].set_xlabel("Date")
  ax[1].set_xlim(pretest_start_date, end_date)
  ax[1].set_ylabel("Treatment - Control")

  if design.is_crossover:
    test_period_weeks = (
        design.runtime_weeks - 2 * design.crossover_washout_weeks) // 2
    test_period_1_start = start_date + dt.timedelta(
        weeks=design.crossover_washout_weeks
    )
    crossover = test_period_1_start + dt.timedelta(weeks=test_period_weeks)
    test_period_2_start = crossover + dt.timedelta(
        weeks=design.crossover_washout_weeks
    )
    ax[0].axvspan(
        start_date, test_period_1_start, color="k", alpha=0.1, label="Washout"
    )
    ax[0].axvspan(crossover, test_period_2_start, color="k", alpha=0.1)
    ax[0].axvline(start_date, color="k", label="Experiment Start")
    ax[0].axvline(crossover, color="k", ls="--", label="Crossover")
    ax[1].axvspan(
        start_date, test_period_1_start, color="k", alpha=0.1, label="Washout"
    )
    ax[1].axvspan(crossover, test_period_2_start, color="k", alpha=0.1)
    ax[1].axvline(start_date, color="k", label="Experiment Start")
    ax[1].axvline(crossover, color="k", ls="--", label="Crossover")
  else:
    ax[0].axvline(start_date, color="k", label="Experiment Start")
    ax[1].axvline(start_date, color="k", label="Experiment Start")

  handles, labels = ax[0].get_legend_handles_labels()
  handles1, labels1 = ax[1].get_legend_handles_labels()
  index_of_delta_line = labels1.index("Treatment - Control")
  index_of_delta_line_ci = labels1.index(
      f"{1.0 - design.alpha:.0%} Confidence Interval"
  )
  handles.append(handles1[index_of_delta_line])
  labels.append(labels1[index_of_delta_line])
  handles.append(handles1[index_of_delta_line_ci])
  labels.append(labels1[index_of_delta_line_ci])
  ax[0].legend(handles, labels, loc="upper left", bbox_to_anchor=(1, 1))

  return fig, ax
