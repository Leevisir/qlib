# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Tuple
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.optimize.zeros import results_c

from ...model.base import ModelFT
from ...data.dataset import DatasetH
from ...data.dataset.handler import DataHandlerLP
from ...model.interpret.base import LightGBMFInt
from sklearn import linear_model

class HFRegressionModel(ModelFT, LightGBMFInt):
    """LightGBM Model for high frequency prediction"""

    def __init__(self, loss="mse", **kwargs):
        if loss not in {"mse", "binary"}:
            raise NotImplementedError
        self.params = {"objective": loss, "verbosity": -1}
        self.params.update(kwargs)
        self.model = None

    def _cal_signal_metrics(self, y_test, l_cut, r_cut):
        """
        Calcaute the signal metrics by daily level
        """
        up_pre, down_pre = [], []
        up_alpha_ll, down_alpha_ll = [], []
        for date in y_test.index.get_level_values(0).unique():
            df_res = y_test.loc[date].sort_values("pred")
            if int(l_cut * len(df_res)) < 10:
                warnings.warn("Warning: threhold is too low or instruments number is not enough")
                continue
            top = df_res.iloc[: int(l_cut * len(df_res))]
            bottom = df_res.iloc[int(r_cut * len(df_res)) :]

            down_precision = len(top[top[top.columns[0]] < 0]) / (len(top))
            up_precision = len(bottom[bottom[top.columns[0]] > 0]) / (len(bottom))

            down_alpha = top[top.columns[0]].mean()
            up_alpha = bottom[bottom.columns[0]].mean()

            up_pre.append(up_precision)
            down_pre.append(down_precision)
            up_alpha_ll.append(up_alpha)
            down_alpha_ll.append(down_alpha)

        return (
            np.array(up_pre).mean(),
            np.array(down_pre).mean(),
            np.array(up_alpha_ll).mean(),
            np.array(down_alpha_ll).mean(),
        )

    def hf_signal_test(self, dataset: DatasetH, threhold=0.2):
        """
        Test the sigal in high frequency test set
        """
        if self.model == None:
            raise ValueError("Model hasn't been trained yet")
        df_test = dataset.prepare("test", col_set=["feature", "label"], data_key=DataHandlerLP.DK_I)
        df_test.dropna(inplace=True)
        x_test, y_test = df_test["feature"], df_test["label"]
        # Convert label into alpha
        y_test[y_test.columns[0]] = y_test[y_test.columns[0]] - y_test[y_test.columns[0]].mean(level=0)

        res = pd.Series(self.model.predict(x_test.values), index=x_test.index)
        y_test["pred"] = res

        up_p, down_p, up_a, down_a = self._cal_signal_metrics(y_test, threhold, 1 - threhold)
        print("===============================")
        print("High frequency signal test")
        print("===============================")
        print("Test set precision: ")
        print("Positive precision: {}, Negative precision: {}".format(up_p, down_p))
        print("Test Alpha Average in test set: ")
        print("Positive average alpha: {}, Negative average alpha: {}".format(up_a, down_a))

    def _prepare_data(self, dataset: DatasetH):
        df_train, df_valid = dataset.prepare(
            ["train", "valid"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
        )

        # x_train, y_train = df_train["feature"], df_train["label"]
        # x_valid, y_valid = df_train["feature"], df_valid["label"]
        # if y_train.values.ndim == 2 and y_train.values.shape[1] == 1:
        #     l_name = df_train["label"].columns[0]
        #     # Convert label into alpha
        #     df_train["label"][l_name] = df_train["label"][l_name] - df_train["label"][l_name].mean(level=0)
        #     df_valid["label"][l_name] = df_valid["label"][l_name] - df_valid["label"][l_name].mean(level=0)
        #     mapping_fn = lambda x: 0 if x < 0 else 1
        #     df_train["label_c"] = df_train["label"][l_name].apply(mapping_fn)
        #     df_valid["label_c"] = df_valid["label"][l_name].apply(mapping_fn)
        #     x_train, y_train = df_train["feature"], df_train["label_c"].values
        #     x_valid, y_valid = df_valid["feature"], df_valid["label_c"].values
        # else:
        #     raise ValueError("LightGBM doesn't support multi-label training")

        # dtrain = lgb.Dataset(x_train, label=y_train)
        # dvalid = lgb.Dataset(x_valid, label=y_valid)

        # df_train = df_train.rename(columns={'Unnamed: 0': 'datetime', 'Unnamed: 1': 'instrument'})
        # # df_train = df_train.drop([0, 1])
        # df_valid = df_valid.rename(columns={'Unnamed: 0': 'datetime', 'Unnamed: 1': 'instrument'})
        # df_valid = df_valid.drop([0, 1])

        return df_train, df_valid
    
    def _prepare_multiple_intervals(self, dataset: pd.DataFrame, freq: str) -> pd.DataFrame:

        '''
        Parameters:
        dataset: Pandas.DataFrame
                minute-level
        freq: str
                frequency that the dataset is subsampled to
        -----------
        Returns:
        subsampled_train: Pandas.DataFrame
                    freq min frequency data, subsampled from dataset
        '''
        return dataset
        # raise NotImplementedError

    def _split_x_y(self, dtrain: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        '''
        Parameters:
        dtrain: Pandas.DataFrame
                training set including feature columns and label columns
        -----------
        Returns:
        x: numpy.ndarray
        y: numpy.ndarray
        '''
        # print(dtrain.columns)
        train_y = dtrain['label'].values
        # train_x = dtrain.drop(['datetime', 'instrument', 'label'], axis=1).values
        train_x = dtrain.drop(['label'], axis=1).values
        return train_x, train_y

    def fit(
        self,
        dataset: DatasetH,
        num_boost_round=1000,
        early_stopping_rounds=50,
        verbose_eval=20,
        evals_result=dict(),
        **kwargs
    ):  
        dtrain, dvalid = self._prepare_data(dataset)
        dtrain_one_min  = self._prepare_multiple_intervals(dtrain, "1min")
        dtrain_five_min = self._prepare_multiple_intervals(dtrain, "5min")
        dtrain_ten_min = self._prepare_multiple_intervals(dtrain, "10min")
        dtrain_thirty_min = self._prepare_multiple_intervals(dtrain, "30min")

        reg = linear_model.LinearRegression()
        min_x, min_y = self._split_x_y(dtrain)
        reg.fit(min_x, min_y)

        reg1 = linear_model.LinearRegression()
        one_min_x, one_min_y = self._split_x_y(dtrain_one_min)
        reg1.fit(one_min_x, one_min_y)

        reg5 = linear_model.LinearRegression()
        five_min_x, five_min_y = self._split_x_y(dtrain_five_min)
        reg5.fit(five_min_x, five_min_y)

        reg10 = linear_model.LinearRegression()
        ten_min_x, ten_min_y = self._split_x_y(dtrain_ten_min)
        reg10.fit(ten_min_x, ten_min_y)

        reg30 = linear_model.LinearRegression()
        thirty_min_x, thirty_min_y = self._split_x_y(dtrain_thirty_min)
        reg30.fit(thirty_min_x, thirty_min_y)

        self.model = [reg1, reg5, reg10, reg30]

    def predict(self, dataset):
        if self.model is None:
            raise ValueError("model is not fitted yet!")
        x_test = dataset.prepare("test", col_set="feature", data_key=DataHandlerLP.DK_I)
        # x_test = x_test.drop(['datetime', 'instrument'], axis=1).values
        results = pd.DataFrame()
        labels = self.model[0].predict(x_test.values)
        tmp = []
        for label in labels:
            tmp.append(label[0])

        # labels = []
        # for point in x_test.values:
        #     label = self.model[0].predict(point)
        #     labels.append(label)
        # results['reg1'] = pd.Series(labels, index=x_test.index)
        # labels = []
        # for point in x_test.values:
        #     label = self.model[1].predict(point)
        #     labels.append(label)
        # results['reg5'] = pd.Series(labels, index=x_test.index)
        # labels = []
        # for point in x_test.values:
        #     label = self.model[2].predict(point)
        #     labels.append(label)
        # results['reg10'] = pd.Series(labels, index=x_test.index)
        # labels = []
        # for point in x_test.values:
        #     label = self.model[3].predict(point)
        #     labels.append(label)
        # results['reg30'] = pd.Series(labels, index=x_test.index)
        results['reg1'] = pd.Series(tmp, index=x_test.index)
        labels = self.model[1].predict(x_test.values)
        tmp = []
        for label in labels:
            tmp.append(label[0])
        results['reg5'] = pd.Series(tmp, index=x_test.index)
        labels = self.model[2].predict(x_test.values)
        tmp = []
        for label in labels:
            tmp.append(label[0])
        results['reg10'] = pd.Series(tmp, index=x_test.index)
        labels = self.model[3].predict(x_test.values)
        tmp = []
        for label in labels:
            tmp.append(label[0])
        results['reg30'] = pd.Series(tmp, index=x_test.index)
        return results

    def finetune(self, dataset: DatasetH, num_boost_round=10, verbose_eval=20):
        """
        finetune model

        Parameters
        ----------
        dataset : DatasetH
            dataset for finetuning
        num_boost_round : int
            number of round to finetune model
        verbose_eval : int
            verbose level
        """
        # Based on existing model and finetune by train more rounds
        dtrain, _ = self._prepare_data(dataset)
        self.model = lgb.train(
            self.params,
            dtrain,
            num_boost_round=num_boost_round,
            init_model=self.model,
            valid_sets=[dtrain],
            valid_names=["train"],
            verbose_eval=verbose_eval,
        )
