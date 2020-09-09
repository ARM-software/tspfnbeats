import os
from warnings import filterwarnings
filterwarnings('ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import tensorflow as tf
from enum import Enum
from typing import Dict, Callable, Optional, Tuple, Any
from dataclasses import dataclass, field
from sklearn.model_selection import train_test_split


class Subset(Enum):
    hourly = 'hourly'
    daily = 'daily'
    weekly = 'weekly'
    monthly = 'monthly'
    quarterly = 'quarterly'
    yearly = 'yearly'


@dataclass
class M4Sources:
    train_file_names: Dict[Subset, str] = field(
        default_factory=lambda: {Subset.hourly: 'Hourly-train.csv',
                                 Subset.daily: 'Daily-train.csv',
                                 Subset.weekly: 'Weekly-train.csv',
                                 Subset.monthly: 'Monthly-train.csv',
                                 Subset.quarterly: 'Quarterly-train.csv',
                                 Subset.yearly: 'Yearly-train.csv'})

    test_file_names: Dict[Subset, str] = field(
        default_factory=lambda: {Subset.hourly: 'Hourly-test.csv',
                                 Subset.daily: 'Daily-test.csv',
                                 Subset.weekly: 'Weekly-test.csv',
                                 Subset.monthly: 'Monthly-test.csv',
                                 Subset.quarterly: 'Quarterly-test.csv',
                                 Subset.yearly: 'Yearly-test.csv'})

@dataclass
class M4SourcesLite:
    train_file_names: Dict[Subset, str] = field(
        default_factory=lambda: {Subset.hourly: 'Hourly-train-lite.csv',
                                 Subset.daily: 'Daily-train-lite.csv'})

    test_file_names: Dict[Subset, str] = field(
        default_factory=lambda: {Subset.hourly: 'Hourly-test-lite.csv',
                                 Subset.daily: 'Daily-test-lite.csv'})

@dataclass
class M4Dataset:
    """
        Class for generating data from M4 dataset
        Each instance manages a subset of M4.

        Data generated is composed of:
        - x: inputs of length = h_mult * H
        - y: outputs of length = H
        - w: valid weight flags of outputs of length = H

        where: H is the forecast horizon for a given subset.
               h_mult is a user define hyperparameter
    """
    sources: Any
    h_mult: Optional[float] = None
    data_dir: Optional[str] = None
    load_subset: Optional[Subset] = None

    validation_size: float = 0.3
    periodicity: Dict[Subset, int] = field(
        default_factory=lambda: {Subset.hourly: 24,
                                 Subset.daily: 1,
                                 Subset.weekly: 1,
                                 Subset.monthly: 12,
                                 Subset.quarterly: 4,
                                 Subset.yearly: 1})

    H: Dict[Subset, int] = field(
        default_factory=lambda: {Subset.hourly: 48,
                                 Subset.daily: 14,
                                 Subset.weekly: 13,
                                 Subset.monthly: 18,
                                 Subset.quarterly: 8,
                                 Subset.yearly: 6})

    read_csv: Callable[[str, str, str], pd.DataFrame] = field(
        default=lambda directory, file, index: pd.read_csv((os.path.join(directory, file)), index_col=index))

    train_file_names: Dict[Subset, str] = field(default_factory=lambda: dict())
    test_file_names: Dict[Subset, str] = field(default_factory=lambda: dict())

    train_df: Optional[pd.DataFrame] = None
    validation_df: Optional[pd.DataFrame] = None
    test_df: Optional[pd.DataFrame] = None

    def __post_init__(self):
        self.train_file_names.update(self.sources.train_file_names)
        self.test_file_names.update(self.sources.test_file_names)

    def update_param(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                if k == 'data_dir':
                    self.data_dir = v
                elif (k == 'load_subset') and isinstance(v, str):
                    setattr(self, k, Subset(v))
                elif isinstance(v, str):
                    setattr(self, k, eval(v))
                else:
                    setattr(self, k, v)
        return self

    def read_source(self, test: bool = False):
        self.full_train_df = self.read_csv(self.data_dir + '/train', self.train_file_names[self.load_subset], 'V1')
        self.train_df, self.validation_df = train_test_split(self.full_train_df, test_size=self.validation_size)
        if test:
            self.test_df = self.read_csv(self.data_dir + '/test', self.test_file_names[self.load_subset], 'V1')
        return self

    def _sample(self, df: pd.DataFrame, n_samp: int, lh: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        future_horizon = self.H[self.load_subset]
        past_horizon = self.h_mult * future_horizon

        data = df.sample(n=n_samp, replace=True, axis=0).to_numpy()
        x = np.zeros((n_samp, past_horizon, 1))
        y = np.zeros((n_samp, future_horizon, 1))
        w = np.zeros_like(y)

        def fn(i, row):
            row = row[~np.isnan(row)]
            if row.shape[0] < 2:
                raise Exception('Empty row')
            anchor = np.random.randint(max(1, row.shape[0] - round(future_horizon * lh)), row.shape[0])
            past = row[max(0, anchor - past_horizon):anchor]
            future = row[anchor:anchor + future_horizon]
            x[i, -past.shape[0]:, 0] = past
            y[i, :future.shape[0], 0] = future
            w[i, :future.shape[0], 0] = 1
            pass

        _ = [fn(i, row) for i, row in enumerate(data)]

        return x, y, w

    def _collect_sample(self, df: pd.DataFrame, n_samples: int, lh: float) \
            -> Dict[str, np.ndarray]:
        x, y, w = self._sample(df, n_samples, lh)
        return {'x': x, 'y': y, 'w': w}

    def _collect_test_sample(self):
        future_horizon = self.H[self.load_subset]
        past_horizon = self.h_mult * future_horizon
        n_samp = self.test_df.shape[0]

        index = self.full_train_df.index
        past_data = self.full_train_df.loc[index].to_numpy()
        future_data = self.test_df.loc[index].to_numpy()

        x = np.zeros((n_samp, past_horizon, 1))
        y = np.zeros((n_samp, future_horizon, 1))
        w = np.zeros_like(y)

        def fn(i, past, future):
            past = past[~np.isnan(past)][-past_horizon:]
            if past.shape[0] < 2:
                raise Exception('Empty row')
            x[i, -past.shape[0]:, 0] = past
            y[i, :future.shape[0], 0] = future
            w[i, :future.shape[0], 0] = 1

        _ = [fn(i, p, f) for i, (p, f) in enumerate(zip(past_data,  future_data))]

        return {'x': x, 'y': y, 'w': w}

    def dataset(self, n_samples: int, lh: float,
                gen_validation: bool = False, gen_p: float = 0.3):

        validation_data = None
        train_data = self._collect_sample(self.train_df, n_samples, lh)
        if gen_validation:
            validation_data = \
                self._collect_sample(self.validation_df, max(1, int(n_samples * gen_p)), lh)

        train_ds = M4Dataset._to_tf_dataset(dict(train_data))
        if gen_validation:
            return train_ds, M4Dataset._to_tf_dataset(dict(validation_data))
        return train_ds

    def dataset_gen(self, n_samples: int, lh: float, gen_validation: bool = False, gen_p: float = 0.3):
        while True:
            train_data = self._collect_sample(self.train_df, n_samples, lh)

            if gen_validation:
                validation_data = \
                    self._collect_sample(self.validation_df, max(1, int(n_samples * gen_p)), lh)
                yield tuple(v for _, v in train_data.items()), tuple(v for _, v in validation_data.items())
            else:
                yield tuple(v for _, v in train_data.items())

    def test_dataset(self) -> tf.data.Dataset:
        data = self._collect_test_sample()
        return M4Dataset._to_tf_dataset(dict(data))

    @staticmethod
    def _to_tf_dataset(data: Dict[str, np.ndarray]) -> tf.data.Dataset:
        tfdata = tf.data.Dataset.from_tensor_slices(
            (
                tf.cast(data['x'], dtype=tf.float32),
                tf.cast(data['y'], dtype=tf.float32),
                tf.cast(data['w'], dtype=tf.float32)
            )
        )

        return tfdata
