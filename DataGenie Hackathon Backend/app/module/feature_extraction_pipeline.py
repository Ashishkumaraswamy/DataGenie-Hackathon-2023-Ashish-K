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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tsa.stattools import adfuller, acf, pacf
from scipy.stats import normaltest

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
        scaler = StandardScaler()  
        X['Value'] = scaler.fit_transform(X['Value'].values.reshape(-1,1))
        mean = np.mean(X['Value'])
        median = np.median(X['Value'])
        # Mode can have multiple values, so take the first one
        mode = X['Value'].mode().values[0]
        variance = np.var(X['Value'])
        std_dev = np.std(X['Value'])
        skewness = skew(X['Value'])
        kurt = kurtosis(X['Value'])
        acf_values = sm.tsa.acf(X['Value'], nlags=5)
        pacf_values = sm.tsa.pacf(X['Value'], nlags=5)
        num_acf_peaks = len([1 for val in acf_values[1:] if abs(val) > 0.025])
        dominant_frequency_acf = acf_values[1:].argmax() + 1
        strength_of_seasonality_acf = acf_values[dominant_frequency_acf]
        has_multiple_seasonal_patterns = len([1 for val in acf_values[1:] if abs(val) > 0.025]) > 1
        x = np.arange(len(X['Value'])).reshape(-1, 1)  # Reshape to 2D array
        linear_reg = LinearRegression().fit(x, X['Value'])
        # Slope (Coefficient)
        slope = linear_reg.coef_[0]

        degree = 2
        polyreg = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        polyreg.fit(x, X['Value'])
        # Coefficients of the polynomial (including higher-order terms)
        poly_coefficients = polyreg.named_steps['linearregression'].coef_
        # print(poly_coefficients)
        # algo = rpt.Pelt(model="rbf").fit(X['Value'])
        # result = algo.predict(pen=10)
        # print("result",result)

        # num_acf_peaks = len([1 for val in acf_values[1:] if abs(val) > 0.025])
        # dominant_frequency_acf_index = acf_values[1:].argmax() + 1
        # dominant_frequency_acf = acf_values[dominant_frequency_acf_index]
        # strength_of_seasonality_acf = dominant_frequency_acf
        # has_multiple_seasonal_patterns = num_acf_peaks > 1

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
        result = sm.tsa.seasonal_decompose(X['Value'], model='additive', period=6)  # Assuming a yearly seasonal period

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

        adf = adfuller(X['Value'])

        lag_acf = acf(X['Value'], nlags=5)
        lag_pacf = pacf(X['Value'], nlags=5, method='ols')

        residuals = X['Value'].diff().dropna()  # Assuming first-order differencing
        

        result_dict = {
            'Mean': mean,
            'Median': median,
            'Mode': mode,
            'Variance': variance,
            'Std Dev': std_dev,
            'Skewness': skewness,
            'Kurtosis': kurt,
            # 'ACF_values' : acf_values,
            # 'PACF_values' : pacf_values,
            # 'Num ACF Peaks': num_acf_peaks,
            # 'Dominant Frequency ACF': dominant_frequency_acf,
            # 'Strength of seasonality ACF':strength_of_seasonality_acf,
            # 'Multiple Seasonal Patterns':has_multiple_seasonal_patterns,
            'Linear Regression Slope': slope,
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

        result_dict['acf_values'] = [float(i) for i in acf_values]
        result_dict['pacf_values'] = [float(i) for i in pacf_values]
        result_dict['num_acf_peaks'] = int(num_acf_peaks)
        result_dict['dominant_frequency'] = float(dominant_frequency_acf)
        result_dict['strength_of_seasonality_acf'] = float(strength_of_seasonality_acf)
        result_dict['has_multiple_seasonal_patterns'] = has_multiple_seasonal_patterns
        result_dict['poly_coefficients'] = [float(i) for i in poly_coefficients]
        _, result_dict['residuals_normality_pvalue'] = normaltest(residuals)
        result_dict['residuals_normality_pvalue'] = float(result_dict['residuals_normality_pvalue'])
        result_dict['stationary'] = bool(adf[1] <= 0.05)  # Check if p-value is less than 0.05
        result_dict['significant_acf_peaks'] = int(sum(np.abs(lag_acf) > 0.25))
        result_dict['significant_pacf_peaks'] = int(sum(np.abs(lag_pacf) > 0.25))
        result_dict['lag1_correlation'] = float(X['Value'].corr(X['Value'].shift(1)))
        result_features = {}
        for key, value in result_dict.items():
            if isinstance(value, list):
                for idx, item in enumerate(value):
                    new_key = f'{key}_{idx}'
                    result_features[new_key] = float(item)
            else:
                result_features[key] = value

        # Return the results as a Pandas Series
        result  = pd.Series(result_features)
        result = result.replace([np.inf, -np.inf], [np.finfo(np.float64).max, np.finfo(np.float64).min]).fillna(0)

        return result

#Create a pipeline
feature_extraction_pipeline = Pipeline([
    ('calculate_stats', FeatureExtractionTransformer())
])

joblib.dump(feature_extraction_pipeline,'feature_extractor_pipeline.pkl')

