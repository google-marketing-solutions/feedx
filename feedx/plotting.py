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

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
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
    histtype: str = "bar",
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
    histtype: the type of histogram to display
  """
  x_label = x_label or column
  y_label = y_label or "Number of Items"

  ax.hist(
      data[column],
      bins="auto",
      histtype=histtype,
      alpha=0.5,
      color=color,
      label=label,
      edgecolor=edgecolor,
  )
  ax.set_title(f"Density Plot of {x_label} Distribution")
  ax.set_xlabel(f"{x_label}")
  ax.set_ylabel(f"{y_label}")


def _make_cumulative_distribution_plot(
    values: np.ndarray,
    ax: plt.Axes,
    x_label: str = "Metric average per item",
    y_label: str = "Percentage of items (number of items)\non a logarithmic scale",
    color: str = "C0",
    label: str = "",
) -> None:
  """Add a cumulative distribution plot of the values provided to the axis.

  This plots the fraction of items with that value or higher as a function of
  the values.

  Args:
    values: Values to be plotted.
    ax: Axes to contain the plot
    x_label: The label for the x-axis.
    y_label: The label for the y-axis.
    color: The color of the line.
    label: The label of the line for a legend.
  """
  values = np.sort(values)
  y_number = len(values) - np.arange(len(values))
  y_total = len(values)
  y_percentage = y_number / y_total

  ax.plot(values, y_percentage, f".-{color}", label=label, lw=1.5)

  ax.set_yscale("log")
  ax.set_xlabel(x_label)
  ax.set_ylabel(y_label)
  ax.yaxis.set_major_formatter(
      mtick.FuncFormatter(
          lambda x, pos: f"{x * 100:.3g}% ({round(x * y_total):,.0f})"
      )
  )


def _make_lorenz_plot(
    values: np.ndarray,
    ax: plt.Axes,
    x_label: str = "Cumulative share of items",
    y_label: str = "Cumulative share of metric",
    color: str = "C0",
    label: str = "",
) -> None:
  """Add a lorenze plot of the values provided to the axis.

  This plots the fraction of the metric that is captured by the top x% of the
  items. More details at https://en.wikipedia.org/wiki/Lorenz_curve.

  Args:
    values: Values to be plotted.
    ax: Axes to contain the plot
    x_label: The label for the x-axis.
    y_label: The label for the y-axis.
    color: The color of the line.
    label: The label of the line for a legend.
  """

  values = np.sort(values)[::-1]
  x_percentage = (np.arange(len(values)) + 1) / len(values)
  y_percentage = np.cumsum(values) / np.sum(values)

  ax.plot(x_percentage, y_percentage, f".-{color}", label=label, lw=1.5)

  ax.set_xscale("log")
  ax.set_xlabel(x_label)
  ax.set_ylabel(y_label)
  ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
  ax.xaxis.set_major_formatter(
      mtick.FuncFormatter(lambda x, pos: f"{x * 100:.3g}%")
  )


def plot_metric_history(
    data: pd.DataFrame,
    metric_column: str,
    item_id_column: str = "item_id",
    week_start_column_name: str = "week_start",
) -> np.ndarray:
  """Plots the weekly distribution and the average of the metric per item per week.

  Args:
    data: data set to plot containing the week_start column and the metrics that
      will be evaluated
    metric_column: metric chosen to run the plot
    item_id_column: the column containing the item id
    week_start_column_name: column containing the week start date

  Returns:
    The axes containing the plots of the metric performance history
  """

  _, axs = plt.subplots(
      nrows=1, ncols=3, figsize=(15, 5), constrained_layout=True
  )

  clean_metric_name = metric_column.replace("_", " ")

  metric_time = (
      data.groupby(week_start_column_name)[[metric_column]].sum().reset_index()
  )
  axs[0].plot(
      metric_time[week_start_column_name],
      metric_time[metric_column],
      marker=".",
      lw=1.5,
  )
  axs[0].set_title(f"Total {clean_metric_name} per week")
  axs[0].set(xlabel="Start Week", ylabel=f"Total {clean_metric_name}")
  axs[0].tick_params(axis="x", rotation=30)

  metric_per_item = data.groupby(item_id_column)[metric_column].mean()

  _make_cumulative_distribution_plot(
      metric_per_item.values,
      axs[1],
      x_label=f"Average {clean_metric_name} per item per week",
      y_label="Percentage of items (number of items)",
  )
  axs[1].set_title(
      f"Percentage of items with this many {clean_metric_name} or more\non"
      " average per week"
  )

  _make_lorenz_plot(
      metric_per_item.values,
      axs[2],
      x_label="Percentage of items",
      y_label=f"Percentage of {clean_metric_name}",
  )
  axs[2].set_title(
      f"Lorenz curve\nWhat percentage of {clean_metric_name} are\ncaptured by"
      " the top x% of items."
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
      edgecolor="black",
      histtype="stepfilled",
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
      edgecolor="black",
      histtype="stepfilled",
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

  is_test_or_test_1 = data["time_period"].isin(["test", "test_1", "washout_1"])
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

  delta.plot(
      lw=1, ax=ax[1], color="C2", marker=".", label="Treatment - Control"
  )
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
        design.runtime_weeks - 2 * design.crossover_washout_weeks
    ) // 2
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


def _plot_horizontal_error_bar_with_infs(
    ax: plt.Axes,
    y_position: list[float],
    value: list[float],
    value_lb: list[float],
    value_ub: list[float],
    errorbar_style: str | None = None,
    **kwargs,
) -> None:
  """Plots the result ranges for each metric.

  This function handles potential infinities that may appear in the results from
  an experiment.

  Args:
    ax: axes that will contain the plot.
    y_position: array containing the vertical position to place the error bars.
    value: array of the results from the experiment for each metric
    value_lb: the lower bound for each result (value)
    value_ub: the upper bound for each result (value)
    errorbar_style: value specifying the style of the errorbar
    **kwargs: additional arguments for the errorbar
  """
  if len(y_position) == 0:
    return

  y_position = np.asarray(y_position)
  value = np.asarray(value)
  value_lb = np.asarray(value_lb)
  value_ub = np.asarray(value_ub)

  inf_lb = ~np.isfinite(value_lb)
  inf_ub = ~np.isfinite(value_ub)

  value = np.copy(value)
  value[~np.isfinite(value)] = 0.0  # If unknown, assume 0 impact

  value_err_lb = value - value_lb
  value_err_ub = value_ub - value
  value_err_lb[inf_lb] = (
      0.0  # set to 0. Infinite error bars are plotted separately
  )
  value_err_ub[inf_ub] = 0.0

  value_lb_max = np.min(value_lb[~inf_lb])
  value_ub_max = np.max(value_ub[~inf_ub])

  eb = ax.errorbar(
      y=y_position, x=value, xerr=[value_err_lb, value_err_ub], **kwargs
  )
  if errorbar_style is not None:
    eb[-1][0].set_linestyle(errorbar_style)

  kwargs["ms"] = 0
  if np.sum(inf_lb) > 0:
    eb = ax.errorbar(
        y=y_position[inf_lb],
        x=value[inf_lb],
        xerr=value[inf_lb] - value_lb_max,
        xuplims=True,
        **kwargs,
    )
    if errorbar_style is not None:
      eb[-1][0].set_linestyle(errorbar_style)

  if np.sum(inf_ub) > 0:
    eb = ax.errorbar(
        y=y_position[inf_ub],
        x=value[inf_ub],
        xerr=value_ub_max - value[inf_ub],
        xlolims=True,
        **kwargs,
    )
    if errorbar_style is not None:
      eb[-1][0].set_linestyle(errorbar_style)


def plot_effects(
    data: pd.DataFrame,
    design: experiment_design.ExperimentDesign,
    relative_difference_column_name: str = "relative_difference",
    relative_difference_lower_bound_column_name: str = "relative_difference_lower_bound",
    relative_difference_upper_bound_column_name: str = "relative_difference_upper_bound",
    is_significant_column_name: str = "is_significant",
) -> np.ndarray:
  """Plots the effects of the experiment for each metric provided.

  This function displays the confidence intervals per metric and indicates which
  results are statistically significant.

  Args:
    data: Experiment results containing the point estimate of the relative
      differences, their upper and lower bounds, and if the results are
      significant for each metric provided.
    design: The design of the experiment.
    relative_difference_column_name: Name of the column containing the point
      estimated of the relative difference between the two samples.
    relative_difference_lower_bound_column_name: Name of the column containing
      the lower bound of the relative difference between the two samples.
    relative_difference_upper_bound_column_name: Name of the column containing
      the upper bound of the relative difference between the two samples.
    is_significant_column_name: If the test is statistically significant for the
      given metric.

  Returns:
    A plot showing the experiment result of the primary and additional metrics

  Raises:
    ValueError: If the primary metric is missing.
  """
  # Identify the primary metric in experiment results using the selected design
  metrics = data.reset_index(names="metric_")["metric_"]
  is_primary_metric = (
      metrics.str.lower() == design.primary_metric.lower()
  ).values

  # If there is no primary metric in the experiment results, raise error
  if np.sum(is_primary_metric) != 1:
    raise ValueError(
        f"Primary metric from design {design.primary_metric.lower()} missing"
        " from data."
    )

  x = metrics.loc[~is_primary_metric].astype("category").cat.codes.values
  x_labels = (
      metrics.loc[~is_primary_metric].astype("category").cat.categories.values
  )

  y = data.loc[~is_primary_metric, relative_difference_column_name].values
  y_lb = data.loc[
      ~is_primary_metric, relative_difference_lower_bound_column_name
  ].values
  y_ub = data.loc[
      ~is_primary_metric, relative_difference_upper_bound_column_name
  ].values

  is_significant = data.loc[
      ~is_primary_metric, is_significant_column_name
  ].values

  x_labels_primary = metrics.loc[is_primary_metric]
  y_primary = data.loc[
      is_primary_metric, relative_difference_column_name
  ].values
  y_primary_lb = data.loc[
      is_primary_metric, relative_difference_lower_bound_column_name
  ].values
  y_primary_ub = data.loc[
      is_primary_metric, relative_difference_upper_bound_column_name
  ].values
  primary_is_significant = data.loc[
      is_primary_metric, is_significant_column_name
  ].values[0]

  _, axs = plt.subplots(
      nrows=2,
      sharex=True,
      height_ratios=[1, len(metrics) - 1],
      figsize=(10, 4),
      constrained_layout=True,
  )

  if primary_is_significant:
    _plot_horizontal_error_bar_with_infs(
        ax=axs[0],
        y_position=[0],
        value=y_primary,
        value_lb=y_primary_lb,
        value_ub=y_primary_ub,
        fmt=".C1",
    )
  else:
    _plot_horizontal_error_bar_with_infs(
        ax=axs[0],
        y_position=[0],
        value=y_primary,
        value_lb=y_primary_lb,
        value_ub=y_primary_ub,
        errorbar_style="--",
        fmt=".C1",
        alpha=0.3,
    )

  axs[0].set_yticks([0], x_labels_primary)
  axs[0].set_title("Primary metric")
  axs[0].grid(axis="x")
  axs[0].axvline(0.0, color="k", lw=1)

  _plot_horizontal_error_bar_with_infs(
      ax=axs[1],
      y_position=x[is_significant],
      value=y[is_significant],
      value_lb=y_lb[is_significant],
      value_ub=y_ub[is_significant],
      fmt=".C0",
  )
  _plot_horizontal_error_bar_with_infs(
      ax=axs[1],
      y_position=x[~is_significant],
      value=y[~is_significant],
      value_lb=y_lb[~is_significant],
      value_ub=y_ub[~is_significant],
      errorbar_style="--",
      fmt=".C0",
      alpha=0.3,
  )

  axs[1].set_yticks(np.arange(len(x_labels)), x_labels)
  axs[1].axvline(0.0, color="k", lw=1)
  axs[1].set_xlabel("Relative Effect Size")
  axs[1].xaxis.set_major_formatter(
      mtick.FuncFormatter(lambda x, pos: f"{x:+.1%}")
  )
  axs[1].set_title("Secondary metrics")
  axs[1].grid(axis="x")

  # Add legend
  legend_lines = [
      Line2D([0], [0], color="k", lw=1),
      Line2D([0], [0], color="k", lw=1, ls="--", alpha=0.3),
  ]
  axs[0].legend(
      legend_lines,
      ["True", "False"],
      title="Is statistically significant?",
      loc="upper left",
      bbox_to_anchor=(1, 1.5),
  )

  return axs
