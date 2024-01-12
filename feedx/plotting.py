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


def _make_density_plot(
    data: pd.DataFrame,
    column: str,
    ax: plt.Axes,
    x_label: str | None = None,
    y_label: str | None = None,
) -> None:
  """Make a density plot using matplotlib visualizing distribution of the metric.

  Args:
    data: data for plotting
    column: name of metric column in the data (eg clicks or impressions)
    ax: class containing the plotted data of the data set provided
    x_label: x-axis label. If not provided, column will be displayed
    y_label: y-axis label. If not provided, "Density" will be displayed
  """
  x_label = x_label or column
  y_label = y_label or "Density"

  ax.hist(
      data[column],
      bins="auto",
      histtype="stepfilled",
      alpha=0.5,
      color="C0",
      label=f"{x_label}",
  )
  ax.hist(data[column], bins=100, histtype="step", alpha=1.0, color="C0")
  ax.set_title(f"Density Plot of {x_label} Distribution")
  ax.set_xlabel(x_label)
  ax.set_ylabel(y_label)
  ax.legend(title="Legend")


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
