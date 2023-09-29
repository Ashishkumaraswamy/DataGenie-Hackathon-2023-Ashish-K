import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import skew, kurtosis, entropy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
import statsmodels.api as sm
from scipy.signal import welch
from scipy.stats import pearsonr
import nolds
import pywt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import joblib


# Custom transformer to extract features
class FeatureExtractionTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def flatten_series(self,data,name):
        result_dict = {f'{name}_{i+1}': value for i, value in enumerate(data)}
        return result_dict
    
    def calculate_lag_features(self,time_series_data, lag=1):
        lagged_data = np.roll(time_series_data, lag)
        correlation_coefficient, _ = pearsonr(time_series_data[lag:], lagged_data[lag:])
        return correlation_coefficient

    def transform(self, X):
        mean = np.mean(X['Value'])
        median = np.median(X['Value'])
        # Mode can have multiple values, so take the first one
        mode = X['Value'].mode().values[0]
        variance = np.var(X['Value'])
        std_dev = np.std(X['Value'])
        skewness = skew(X['Value'])
        kurt = kurtosis(X['Value'])

        # Calculate percentiles (e.g., 25th, 75th percentiles)
        percentiles = np.percentile(X['Value'], [25, 75])

        # Calculate autocorrelation at different lags (e.g., lag 1, lag 2)
        autocorr_lag1 = X['Value'].autocorr(lag=1)
        autocorr_lag2 = X['Value'].autocorr(lag=2)

        # Calculate rolling statistics (e.g., rolling mean and rolling standard deviation)
        rolling_window = 3  # Adjust the window size as needed
        rolling_mean = X['Value'].rolling(window=rolling_window).mean()
        rolling_std_dev = X['Value'].rolling(window=rolling_window).std()

        # Extract seasonal, trend, and residual components for each month
        result = sm.tsa.seasonal_decompose(X['Value'], model='additive', period=12)  # Assuming a yearly seasonal period

        # Extract Seasonal Components
        trend_component = result.trend
        seasonal_component = result.seasonal
        residual_component = result.resid

        seasonal_result = result.seasonal
        # Calculate Seasonal Indices
        seasonal_indices = [seasonal_component[i] for i in range(12)]  # Assuming a yearly seasonal period

        # Determine the presence or absence of seasonality
        seasonality_present = any(abs(seasonal_component) > 0)

        fft_values = np.fft.fft(X['Value'])
        frequencies = np.fft.fftfreq(len(X['Value']))

        # Calculate Power Spectral Density (PSD) using Welch's method
        frequencies_welch, psd = welch(X['Value'], nperseg=len(X['Value']))

        denominator = np.sum(psd[(frequencies_welch >= 4) & (frequencies_welch <= 8)])
        if denominator != 0:
            frequency_delta_theta_ratio = np.sum(psd[(frequencies_welch >= 1) & (frequencies_welch <= 4)]) / denominator
        else:
            frequency_delta_theta_ratio = 0  # or any default value you want to assign

        first_differences = np.diff(X['Value'])

        # Calculate Lag-based Feature (e.g., with lag=1)
        lag_feature = self.calculate_lag_features(X['Value'], lag=1)

        information_entropy = nolds.sampen(X['Value'])

        # Calculate Sample Entropy
        sample_entropy = nolds.sampen(X['Value'])

        # Perform Discrete Wavelet Transform (DWT)
        coeffs = pywt.dwt(X['Value'], 'db1')

        # Extract coefficients from DWT
        cA, cD = coeffs  # cA: Approximation coefficients, cD: Detail coefficients

        result_dict = {
            'Mean': mean,
            'Median': median,
            'Mode': mode,
            'Variance': variance,
            'Std Dev': std_dev,
            'Skewness': skewness,
            'Kurtosis': kurt,
            '25th Percentile': percentiles[0],
            '75th Percentile': percentiles[1],
            'Auto_Corr_Lag1': autocorr_lag1,
            'Auto_Corr_Lag2': autocorr_lag2,
            'Rolling_Mean': rolling_mean.iloc[-1],
            'Rolling_Std_Dev': rolling_std_dev.iloc[-1],
            "Seasonality Present": seasonality_present,
            "Seasonal Mean": np.mean(seasonal_component),
            "Seasonal Variance": np.var(seasonal_component),
            "Seasonal Max" : np.max(seasonal_component),
            "Seasonal Min" : np.min(seasonal_component),
            "Seasonal 25th percentile" : np.percentile(seasonal_component, 25),
            "Seasonal 75th percentile" : np.percentile(seasonal_component, 75),
            "Trend Mean" : np.mean(trend_component),
            "Trend Variance" : np.var(trend_component),
            "Trend Max" : np.max(trend_component),
            "Trend Min" : np.min(trend_component),
            "Trend 25th percentile" : np.percentile(trend_component, 25),
            "Trend 75th percentile" : np.percentile(trend_component, 75),
            "Residual Mean" : np.mean(residual_component),
            "Residual Variance" : np.var(residual_component),
            "Residual Max" : np.max(residual_component),
            "Residual Min" : np.min(residual_component),
            "Residual 25th percentile" : np.percentile(residual_component, 25),
            "Residual 75th percentile" : np.percentile(residual_component, 75),
            "Seasonal Indices Mean" : np.mean(seasonal_indices),
            "Seasonal Indices Variance" : np.var(seasonal_indices),
            "Seasonal Indices Max" : np.max(seasonal_indices),
            "Seasonal Indices Min" : np.min(seasonal_indices),
            "Seasonal Indices 25th percentile" : np.percentile(seasonal_indices, 25),
            "Seasonal Indices 75th percentile" : np.percentile(seasonal_indices, 75),
            "Mean Frequency" : np.mean(frequencies),
            "Median Frequency" : np.median(frequencies),
            "Frequency Variance" : np.var(frequencies),
            "Frequency Skewness" : skew(frequencies),
            "Frequency Kurtosis" : kurtosis(frequencies),
            "Frequency 25th Percentile" : np.percentile(frequencies, 25),
            "Frequency 75th Percentile" : np.percentile(frequencies, 75),
            "Fequency Entropy Value" : entropy(frequencies),
            "Fequency Peak Value" : frequencies[np.argmax(np.abs(fft_values))],
            "Frequency Delta Theta Ratio" : frequency_delta_theta_ratio,
            "Lag 1 Feature Correlation" : lag_feature,
            "First Order Difference Mean" : np.mean(first_differences),
            "First Order Difference Median" : np.median(first_differences),
            "First Order Difference Variance" : np.var(first_differences),
            "First Order Difference Skewness" : skew(first_differences),
            "First Order Difference Kurtosis" : kurtosis(first_differences),
            "First Order Difference 25th Percentile" : np.percentile(first_differences, 25),
            "First Order Difference 75th Percentile" : np.percentile(first_differences, 75),
            "First Order Difference Entropy" : entropy(first_differences),
            "Information Entropy" : information_entropy,
            "Sample Entropy" : sample_entropy,
            "cA Coeffecients Mean" : np.mean(cA),
            "cA Coeffecients Variance" : np.var(cA),
            "cA Coeffecients Max" : np.max(cA),
            "cA Coeffecients Min" : np.min(cA),
            "cA Coeffecients 25th percentile" : np.percentile(cA, 25),
            "cA Coeffecients 75th percentile" : np.percentile(cA, 75),
            "cD Coeffecients Mean" : np.mean(cD),
            "cD Coeffecients Variance" : np.var(cD),
            "cD Coeffecients Max" : np.max(cD),
            "cD Coeffecients Min" : np.min(cD),
            "cD Coeffecients 25th percentile" : np.percentile(cD, 25),
            "cD Coeffecients 75th percentile" : np.percentile(cD, 75),
        }

        # Return the results as a Pandas Series
        result  = pd.Series(result_dict)
        result = result.replace([np.inf, -np.inf], [np.finfo(np.float64).max, np.finfo(np.float64).min]).fillna(0)

        return result

#Create a pipeline
feature_extraction_pipeline = Pipeline([
    ('calculate_stats', FeatureExtractionTransformer())
])

joblib.dump(feature_extraction_pipeline,'feature_extractor_pipeline.pkl')

