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

"""Plot historical and simulation data."""

import altair as alt
import pandas as pd

COLORS = [
    "#4285F4",
    "#EA4335",
    "#FBBC04",
    "#34A853",
    "#185ABC",
    "#B31412",
    "#EA8600",
    "#137333"
]


def make_density_plot(
    data: pd.DataFrame,
    column: str,
    x_label: str | None = None,
    y_label: str | None = None,
    color_id: int = 0,
):
  """Make a density plot using altair visualizing distribution of the metric.
  
  Args:
    data: synthetic, historical, or experiment data
    column: name of metric column in the data (eg clicks or impressions)
    x_label: x-axis label. If not provided, column will be displayed
    y_label: y-axis label. If not provided, "Density" will be displayed
    color_id: ID of selected color to display in the plot
    
  Returns:
    Plot displaying proportion of items for a given number of clicks.
  """

  x_label = x_label or column
  y_label = y_label or "Density"

  density_base = alt.Chart(data).transform_density(
      column,
      as_=[x_label, y_label],
  )

  area = density_base.mark_area(opacity=0.3, color=COLORS[color_id])
  line = density_base.mark_line(color=COLORS[color_id])

  combined = (area + line).encode(
      x=f'{x_label}:Q',
      y=f'{y_label}:Q',
  )

  return combined
