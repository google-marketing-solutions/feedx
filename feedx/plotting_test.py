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

"""Tests for plotting."""

import unittest
import numpy as np
import pandas as pd

from feedx import plotting


class MakeDensityPlot(unittest.TestCase):

  def test_make_density_plot(self):
    data_plot = pd.DataFrame({
        "date": (
            ["2023-10-02"] * 125
            + ["2023-10-09"] * 125
            + ["2023-10-16"] * 125
            + ["2023-10-23"] * 125
        ),
        "item_id": np.linspace(0, 1, 500),
        "impressions": np.linspace(0, 10, 500),
        "clicks": np.linspace(0, 10, 500),
    })
    data_plot["date"] = pd.to_datetime(data_plot["date"])
    data_plot["impressions"] = data_plot["impressions"].astype(int)
    data_plot["clicks"] = data_plot["clicks"].astype(int)

    column = "clicks"
    x_label = "Average clicks per week"
    y_label = "Density"
    color_id = 1
    plot = plotting.make_density_plot(
        data=data_plot, column=column, x_label=x_label, y_label=y_label,
        color_id=color_id)
    assert len(plot["data"]) == 500


if __name__ == "__main__":
  unittest.main()
