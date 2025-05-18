import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
from scipy.stats import pearsonr

# 忽略警告
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_and_prepare_data(file_path):
    """加载并预处理数据"""
    # 加载数据
    df = pd.read_csv(file_path)

    # 基本预处理
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').set_index('date')

    # 使用AQI指数作为目标变量
    aqi_series = df['AQI_index']

    return aqi_series


def difference_series(series, d=1):
    """对序列进行差分"""
    return series.diff(d).dropna()


def plot_acf_pacf(series, lags=30):
    """绘制自相关和偏自相关图"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(series, lags=lags, ax=ax1)
    plot_pacf(series, lags=lags, ax=ax2)
    plt.tight_layout()
    plt.show()


def calculate_mape(actual, predicted):
    """计算平均绝对百分比误差(MAPE)"""
    actual, predicted = np.array(actual), np.array(predicted)
    # 避免除以零的情况
    non_zero_mask = actual != 0
    return np.mean(np.abs((actual[non_zero_mask] - predicted[non_zero_mask]) / actual[non_zero_mask])) * 100


def calculate_lagged_correlation(actual, predicted, max_lag=5):
    """
    计算预测值与真实值之间的滞后相关系数
    参数:
        actual: 真实值序列
        predicted: 预测值序列
        max_lag: 最大滞后阶数
    返回:
        lags: 滞后阶数数组
        correlations: 各滞后阶数对应的相关系数
        p_values: 各相关系数的p值
    """
    lags = range(-max_lag, max_lag + 1)
    correlations = []
    p_values = []

    for lag in lags:
        if lag < 0:
            # 负滞后：预测值领先于真实值
            corr, pval = pearsonr(actual[-lag:], predicted[:lag])
        elif lag > 0:
            # 正滞后：预测值滞后于真实值
            corr, pval = pearsonr(actual[:-lag], predicted[lag:])
        else:
            # 零滞后：同期相关
            corr, pval = pearsonr(actual, predicted)

        correlations.append(corr)
        p_values.append(pval)

    return lags, correlations, p_values


def plot_lagged_correlation(lags, correlations, p_values, threshold=0.05):
    """绘制滞后相关系数图"""
    plt.figure(figsize=(12, 6))

    # 标记统计显著的相关系数
    significant = [p < threshold for p in p_values]

    # 绘制条形图
    bars = plt.bar(lags, correlations, color=['red' if sig else 'blue' for sig in significant])

    # 添加标签和标题
    plt.title('预测值与真实值的滞后相关系数', fontsize=14)
    plt.xlabel('滞后阶数', fontsize=12)
    plt.ylabel('相关系数', fontsize=12)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.grid(True, linestyle='--', alpha=0.5)

    # 添加图例
    red_patch = plt.Rectangle((0, 0), 1, 1, fc='red', label='显著相关 (p<0.05)')
    blue_patch = plt.Rectangle((0, 0), 1, 1, fc='blue', label='不显著相关')
    plt.legend(handles=[red_patch, blue_patch])

    plt.tight_layout()
    plt.show()


def train_arima_model(series, order, test_size=0.2):
    """训练ARIMA模型并评估"""
    # 划分训练集和测试集
    split_idx = int(len(series) * (1 - test_size))
    train, test = series[:split_idx], series[split_idx:]

    history = [x for x in train]
    predictions = []

    # 逐步预测
    for t in range(len(test)):
        model = ARIMA(history, order=order)
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history.append(test[t])

    # 计算评估指标
    mse = mean_squared_error(test, predictions)
    rmse = sqrt(mse)
    mae = mean_absolute_error(test, predictions)
    r2 = r2_score(test, predictions)
    mape = calculate_mape(test, predictions)  # 计算MAPE

    # 计算滞后相关系数
    lags, correlations, p_values = calculate_lagged_correlation(test, predictions)

    print("\n模型评估指标:")
    print(f"均方误差(MSE): {mse:.2f}")
    print(f"均方根误差(RMSE): {rmse:.2f}")
    print(f"平均绝对误差(MAE): {mae:.2f}")
    print(f"平均绝对百分比误差(MAPE): {mape:.2f}%")
    print(f"R²分数: {r2:.4f}")

    # 打印滞后相关系数
    print("\n滞后相关系数分析:")
    for lag, corr, pval in zip(lags, correlations, p_values):
        print(f"滞后 {lag:2d}: 相关系数={corr:.4f} (p值={'<0.05' if pval < 0.05 else f'{pval:.4f}'})")

    # 绘制预测结果
    plt.figure(figsize=(12, 6))
    plt.plot(test.index, test, label='实际值')
    plt.plot(test.index, predictions, color='red', label='预测值')
    plt.title('ARIMA模型预测结果')
    plt.xlabel('日期')
    plt.ylabel('AQI指数')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # 绘制滞后相关系数图
    plot_lagged_correlation(lags, correlations, p_values)

    return model_fit, predictions


def main():
    print("============== 广州AQI指数ARIMA模型 ==============")

    # 1. 加载和准备数据
    file_path = r'data\datas_guangzhou.csv'
    aqi_series = load_and_prepare_data(file_path)

    # 3. 可视化原始数据
    plt.figure(figsize=(12, 6))
    aqi_series.plot()
    plt.title('广州AQI指数原始数据')
    plt.xlabel('日期')
    plt.ylabel('AQI指数')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # 5. 训练ARIMA模型
    print("\n训练ARIMA模型...")
    # 根据ACF和PACF图以及差分结果选择(p,d,q)参数
    order = (1, 0, 1)  # 根据分析结果，数据已经是平稳的，所以d=0

    model_fit, predictions = train_arima_model(aqi_series, order)

    # 6. 输出模型摘要
    print("\n模型摘要:")
    print(model_fit.summary())


if __name__ == "__main__":
    main()