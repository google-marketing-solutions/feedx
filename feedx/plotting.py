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
import pandas as pd


def _make_density_plot(
    data: pd.DataFrame,
    column: str,
    ax: plt.Axes,
    x_label: str | None = None,
    y_label: str | None = None
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
      data[column], bins="auto", histtype="stepfilled", alpha=0.5, color="C0",
      label=f'{x_label}')
  ax.hist(data[column], bins=100, histtype="step", alpha=1.0, color="C0")
  ax.set_title(f'Density Plot of {x_label} Distribution')
  ax.set_xlabel(x_label)
  ax.set_ylabel(y_label)
  ax.legend(title="Legend")
