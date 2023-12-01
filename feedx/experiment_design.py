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

"""Define experiment design for running A/B tests on Shopping feeds."""

import dataclasses
import hashlib
import numpy as np
import pandas as pd
import yaml


@dataclasses.dataclass
class ExperimentDesign:
  """An experiment design.

  Attributes:
    n_items_before_trimming: The number of items before any trimming
    n_items_after_pre_trim: The number of items after the pre trimming step.
      This is the number of items that will actually be in the experiment.
    n_items_after_post_trim: The number of items after the post trimming step.
    runtime_weeks: The number of weeks the experiment will last.
    primary_metric: The primary metric to be analyzed during the experiment.
      This is the metric the experiment is optimized for.
    pretest_weeks: The number of weeks of data to look at before experiment
      begins, for either pre-trimming or the cuped adjustment.
    crossover_washout_weeks: The number of weeks to ignore to avoid
      contamination in a crossover experiment. Set to None for a non-crossover
      experiment.
    is_crossover: Is this a crossover experiment design?
    pre_trim_top_percentile: The fraction of items with the highest primary
      metric in the pretest period to remove.
    pre_trim_bottom_percentile: The fraction of items with the lowest primary
      metric in the pretest period to remove.
    post_trim_percentile: The fraction of items with both the highest and lowest
      primary metric in the test runtime period to trim in the analysis.
    alpha: The target false positive rate of the experiment. Defaults to 0.05.
    power: The target power for the experiment. Defaults to 0.8.
    coinflip_salt: The salt used to randomise items into control and treatment.
      None if randomisation has not been done.
  """
  n_items_before_trimming: int
  runtime_weeks: int
  primary_metric: str
  pretest_weeks: int
  is_crossover: bool
  pre_trim_top_percentile: float
  pre_trim_bottom_percentile: float
  post_trim_percentile: float

  crossover_washout_weeks: int | None = None
  alpha: float = 0.05
  power: float = 0.8
  coinflip_salt: str | None = None

  n_items_after_pre_trim: int = dataclasses.field(init=False)
  n_items_after_post_trim: int = dataclasses.field(init=False)

  def _validate_design(self):
    """Performs simple validation on the design parameters."""

    if self.runtime_weeks <= 0:
      raise ValueError("Runtime weeks must be > 0.")
    if self.pretest_weeks < 0:
      raise ValueError("Pretest weeks must be >= 0.")
    if self.pre_trim_top_percentile < 0.0:
      raise ValueError("Pre_trim_top_percentile must be >= 0.0.")
    if self.pre_trim_bottom_percentile < 0.0:
      raise ValueError("Pre_trim_bottom_percentile must be >= 0.0.")
    if self.post_trim_percentile < 0.0:
      raise ValueError("Post_trim_percentile must be >= 0.0.")
    if (self.pre_trim_top_percentile + self.pre_trim_bottom_percentile) >= 1.0:
      raise ValueError(
          "Pre_trim_top_percentile + pre_trim_bottom_percentile must be < 1.0."
      )
    if self.post_trim_percentile >= 0.5:
      raise ValueError("Post_trim_percentile must be < 0.5.")

    has_pretrimming = (self.pre_trim_top_percentile > 0.0) | (
        self.pre_trim_bottom_percentile > 0.0
    )
    if (self.pretest_weeks == 0) & has_pretrimming:
      raise ValueError("Cannot do any pre-trimming if the pretest weeks are 0.")

    if self.is_crossover:
      if self.crossover_washout_weeks is None:
        raise ValueError(
            "Must specify the crossover washout weeks for a crossover test."
        )

      if self.crossover_washout_weeks < 0:
        raise ValueError(
            "Crossover washout weeks must be >= 0 for a crossover test."
        )

      if self.runtime_weeks <= 2 * self.crossover_washout_weeks:
        raise ValueError(
            "Runtime weeks must be > 2 * crossover washout weeks for a"
            " crossover test, otherwise there will be no data left after"
            " excluding the washout weeks."
        )

      if self.runtime_weeks % 2 != 0:
        raise ValueError(
            "Runtime weeks must be even for a crossover test, so that there is"
            " an equal number of weeks in each period of the crossover."
        )

    if self.n_items_after_post_trim <= 50:
      raise ValueError(
          "There must be more than 50 items remaining after all trimming, only"
          f" have {self.n_items_after_post_trim}."
      )

  def __post_init__(self):
    pre_trim_top = int(
        np.floor(self.pre_trim_top_percentile * self.n_items_before_trimming)
    )
    pre_trim_bottom = int(
        np.floor(self.pre_trim_bottom_percentile * self.n_items_before_trimming)
    )
    self.n_items_after_pre_trim = (
        self.n_items_before_trimming - pre_trim_top - pre_trim_bottom
    )

    post_trim_each_side = int(
        np.floor(self.post_trim_percentile * self.n_items_after_pre_trim)
    )
    self.n_items_after_post_trim = (
        self.n_items_after_pre_trim - 2 * post_trim_each_side
    )

    self._validate_design()

  @property
  def design_id(self) -> str:
    """Returns a unique identifier for the design."""

    return hashlib.md5(str(self).encode("utf-8")).hexdigest()

  @classmethod
  def load_from_yaml(cls, file_path: str) -> "ExperimentDesign":
    """Load the experiment design from yaml file.

    Args:
      file_path: The path to a yaml file containing the experiment design.

    Returns:
      The loaded experiment design.

    Raises:
      ValueError: If the number of items after trimming in the yaml file are
        inconsistent with the trim percentiles.
    """
    with open(file_path, "r") as file:
      raw_values = yaml.safe_load(file)

    loaded_n_items_after_pre_trim = raw_values.pop("n_items_after_pre_trim")
    loaded_n_items_after_post_trim = raw_values.pop("n_items_after_post_trim")

    design = cls(**raw_values)

    if design.n_items_after_pre_trim != loaded_n_items_after_pre_trim:
      raise ValueError(
          "n_items_after_pre_trim in the yaml file does not match what is"
          " calculated from the n_items_before_trimming and the trimming"
          f" percentiles. In the yaml file: {loaded_n_items_after_pre_trim},"
          f" expected: {design.n_items_after_pre_trim}."
      )
    if design.n_items_after_post_trim != loaded_n_items_after_post_trim:
      raise ValueError(
          "n_items_after_post_trim in the yaml file does not match what is"
          " calculated from the n_items_before_trimming and the trimming"
          f" percentiles. In the yaml file: {loaded_n_items_after_post_trim},"
          f" expected: {design.n_items_after_post_trim}."
      )

    return design

  def write_to_yaml(self, file_path: str) -> None:
    """Write the experiment design to a yaml file.

    Args:
      file_path: The path to write to.
    """
    raw_values = dataclasses.asdict(self)
    with open(file_path, "w") as file:
      yaml.dump(raw_values, file)


class Coinflip:
  """Coinflip to assign items to control or treatment groups.

  This coinflip is psuedo-random but deterministic, meaning that given the same
  item_id and salt it will always return the same treatment assignment.

  It assigns the items with a 50/50 chance of being in treatment or control.

  When the coinflip is called, it prepends the salt to the argument it is called
  on, and then hashes the result, creating a psuedo-random number.

  Attributes:
    salt: The salt used for the hash function.
  """

  def __init__(self, salt: str):
    """Instantiates the coinflip function with the salt provided.

    Args:
      salt: The salt to use for the hash.

    Raises:
      ValueError: If the salt is an empty string.
    """
    if not salt:
      raise ValueError("Salt must not be empty.")

    self.salt = salt

  @classmethod
  def with_random_salt(cls, salt_length: int = 10) -> "Coinflip":
    rng = np.random.default_rng()
    salt = str(rng.bytes(salt_length).hex())
    return cls(salt)

  def __call__(self, item_id: str) -> int:
    """Assigns each sample to either the control or treatment group.

    Args:
      item_id: item_id of the sample

    Returns:
      Treatment assignment (0 for control or 1 for treatment).
    """
    salted_item_id = self.salt + str(item_id)
    hash_num = hashlib.md5(salted_item_id.encode("utf-8"))
    hex_hash = hash_num.hexdigest()
    int_hash = int(hex_hash, 16)
    return int_hash % 2


def trim_outliers(
    data: pd.DataFrame,
    order_by: str,
    trim_percentile_top: float,
    trim_percentile_bottom: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
  """Trims outliers from the data.

  1. Calculates the number of rows to trim from the top and bottom by
    multiplying the trim percentile by the number of rows in the data, and
    flooring the result.
  2. If 0 rows are to be trimmed, return the data.
  3. Rank the data by the order_by column, with ties being given a random order
    using the random number generator rng.
  4. Remove the rows to be trimmed from either end and return the data.

  Args:
    data: The data to be trimmed.
    order_by: The column to order by to select the outliers to trim.
    trim_percentile_top: The fraction of the highest values to remove from the
      data.
    trim_percentile_bottom: The fraction of the lowest values to remove from the
      data.
    rng: The random number generator used to break ties in the ordering.

  Returns:
    The data with the outliers trimmed.

  Raises:
    ValueError: If either of the trim_perentile_top or trim_percentile_bottom
      are negative, or if trim_perentile_top + trim_percentile_bottom >= 1.0.
      This is to ensure that you do not trim 100% of the data.
  """
  if trim_percentile_top < 0.0:
    raise ValueError("trim_percentile_top must be >= 0.0.")
  if trim_percentile_bottom < 0.0:
    raise ValueError("trim_percentile_bottom must be >= 0.0.")
  if trim_percentile_top + trim_percentile_bottom >= 1.0:
    raise ValueError(
        "trim_percentile_top + trim_percentile_bottom must be < 1.0, otherwise"
        f" you will remove all the data. {trim_percentile_top = }, "
        f"{trim_percentile_bottom = }, so "
        f"{trim_percentile_top + trim_percentile_bottom = } >= 1.0."
    )

  n_samples = len(data.index.values)
  n_trim_top = int(np.floor(n_samples * trim_percentile_top))
  n_trim_bottom = int(np.floor(n_samples * trim_percentile_bottom))

  if (n_trim_top == 0) & (n_trim_bottom == 0):
    return data

  min_rank = n_trim_top
  max_rank = n_samples - n_trim_bottom

  rank = data.sample(frac=1.0, random_state=rng)[order_by].rank(
      method="first", ascending=False
  )

  return data.loc[(rank > min_rank) & (rank <= max_rank)]
