import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted


class AdultPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, numeric, categorical, quantiles=(0.01, 0.99), derived=False):
        self.numeric = numeric
        self.categorical = categorical
        self.quantiles = quantiles
        self.derived = derived

    def _categorical_values(self, frame):
        return frame[self.categorical].astype(object).replace('?', np.nan)

    def _numeric_values(self, frame):
        return frame[self.numeric].replace('?', np.nan).astype(np.float64)

    def _add_features(self, values):
        result = pd.DataFrame(values, columns=self.numeric)
        if self.derived:
            result['capital_net'] = result['capital-gain'] - result['capital-loss']
            result['hours_per_age'] = result['hours-per-week'] / (result['age'] - 16)
        return result

    def fit(self, frame, y=None):
        self.numeric_imputer_ = SimpleImputer(strategy='median')
        values = self.numeric_imputer_.fit_transform(self._numeric_values(frame))
        if values.shape[1] != len(self.numeric):
            raise ValueError('A numeric training column is entirely missing')
        self.lower_, self.upper_ = np.quantile(values, self.quantiles, axis=0, method='linear')
        numeric = self._add_features(np.clip(values, self.lower_, self.upper_))
        self.scaler_ = StandardScaler().fit(numeric)
        self.categorical_imputer_ = SimpleImputer(strategy='most_frequent')
        categories = self.categorical_imputer_.fit_transform(self._categorical_values(frame))
        if categories.shape[1] != len(self.categorical):
            raise ValueError('A categorical training column is entirely missing')
        self.encoder_ = OneHotEncoder(handle_unknown='ignore', sparse_output=True, dtype=np.float64)
        self.encoder_.fit(categories)
        self.numeric_names_ = list(numeric.columns)
        self.fit_row_count_ = len(frame)
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        self.n_features_in_ = len(frame.columns)
        return self

    def clean_frame(self, frame):
        check_is_fitted(self, 'encoder_')
        values = self.numeric_imputer_.transform(self._numeric_values(frame))
        result = self._add_features(np.clip(values, self.lower_, self.upper_))
        result.index = frame.index
        categories = self.categorical_imputer_.transform(self._categorical_values(frame))
        result[self.categorical] = categories
        return result

    def transform(self, frame):
        cleaned = self.clean_frame(frame)
        numeric = self.scaler_.transform(cleaned[self.numeric_names_])
        categories = self.encoder_.transform(cleaned[self.categorical].to_numpy())
        result = sparse.hstack([sparse.csr_matrix(numeric), categories], format='csr')
        if not np.isfinite(result.data).all():
            raise ValueError('Processed values must be finite')
        return result

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, 'encoder_')
        return np.r_[self.numeric_names_, self.encoder_.get_feature_names_out(self.categorical)]
