import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, GRU, Dense, Dropout, BatchNormalization, Bidirectional
EarlyStopping       = tf.keras.callbacks.EarlyStopping
ModelCheckpoint     = tf.keras.callbacks.ModelCheckpoint
ReduceLROnPlateau   = tf.keras.callbacks.ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from datetime import timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import ParameterGrid
# 忽略警告
warnings.filterwarnings('ignore')

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 1. 数据加载
def load_data(file_path):
    """
    加载CSV数据文件

    参数:
        file_path: CSV文件路径
    返回:
        DataFrame: 加载的数据
    """
    print("1. 数据加载")
    # 加载CSV文件
    df = pd.read_csv(r'data\datas_guangzhou.csv')
    print(f"  - 数据加载完成，共 {df.shape[0]} 行，{df.shape[1]} 列")
    return df


# 2. 数据预处理
def preprocess_data(df):
    """
    数据预处理：清洗、转换和特征工程

    参数:
        df: 原始数据DataFrame
    返回:
        DataFrame: 预处理后的数据
    """
    print("\n2. 数据预处理")

    # 2.1 数据类型转换
    print("  - 2.1 转换日期格式")
    df['date'] = pd.to_datetime(df['date'])

    # 2.2 检查并处理缺失值
    print("  - 2.2 处理缺失值")
    missing_values = df.isnull().sum()
    print(f"    缺失值统计:\n{missing_values}")

    # 使用前后值插值填充缺失值
    if missing_values.sum() > 0:
        df = df.interpolate(method='time')
        # 如果首尾有缺失值无法插值，使用最近的有效值填充
        df = df.fillna(method='bfill').fillna(method='ffill')

    # 2.3 异常值检测与处理
    print("  - 2.3 检测和处理异常值")
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        # 计算IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        # 定义异常值边界
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # 统计异常值数量
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        if len(outliers) > 0:
            print(f"    '{col}' 列有 {len(outliers)} 个异常值")

            # 将异常值设置为边界值（剪切法）
            df.loc[df[col] < lower_bound, col] = lower_bound
            df.loc[df[col] > upper_bound, col] = upper_bound

    # 2.4 特征工程
    print("  - 2.4 特征工程")

    # 从日期提取时间特征
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek  # 0=星期一，6=星期日
    df['quarter'] = df['date'].dt.quarter
    df['is_weekend'] = df['dayofweek'].apply(lambda x: 1 if x >= 5 else 0)

    # 季节编码（北半球）
    def get_season(month):
        if month in [12, 1, 2]:
            return '冬季'
        elif month in [3, 4, 5]:
            return '春季'
        elif month in [6, 7, 8]:
            return '夏季'
        else:
            return '秋季'

    df['season'] = df['month'].apply(get_season)

    # 季节的数值编码（用于机器学习模型）
    season_map = {'春季': 1, '夏季': 2, '秋季': 3, '冬季': 4}
    df['season_code'] = df['season'].map(season_map)

    '''### 创建季节性sin-cos特征（对周期性建模有帮助）
    #df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

    # 创建主要污染物的滞后特征（前1天、前7天）
    pollution_cols = ['AQI_index', 'PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']

    for col in pollution_cols:
        # 前1天的值
        df[f'{col}_lag1'] = df[col].shift(1)
        # 前7天的值
        df[f'{col}_lag7'] = df[col].shift(7)
        # 前7天的平均值
        df[f'{col}_rolling7'] = df[col].rolling(window=7).mean()

    # 删除由于创建滞后特征导致的缺失行
    df = df.dropna()
###'''
    print(f"  - 预处理后数据形状: {df.shape}")
    return df


# 3. 探索性数据分析
def exploratory_data_analysis(df):
    """
    探索性数据分析与可视化

    参数:
        df: 预处理后的数据
    """
    print("\n3. 探索性数据分析...")

    # 3.1 基本统计描述
    print("  - 3.1 基本统计描述")
    pollution_cols = ['AQI_index', 'PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
    stats = df[pollution_cols].describe()
    print(stats)

    # 3.2 时间趋势分析
    print("  - 3.2 创建时间趋势图")

    # 选择数值型列进行分析
    pollution_cols = ['AQI_index', 'PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']

    # 创建一个日期索引的数据框，只包含需要的数值列
    trend_data = df[['date'] + pollution_cols].copy()
    trend_data.set_index('date', inplace=True)

    # 确保所有列都是数值类型
    for col in pollution_cols:
        trend_data[col] = pd.to_numeric(trend_data[col], errors='coerce')

    # 按月分组并计算平均值
    try:
        monthly_data = trend_data.resample('M').mean()
        print(f"    月度数据形状: {monthly_data.shape}")

        plt.figure(figsize=(14, 8))
        plt.subplot(2, 1, 1)
        plt.plot(monthly_data.index, monthly_data['AQI_index'], 'b-', label='AQI指数')
        plt.title('广州空气质量指数(AQI)月度趋势', fontsize=14)
        plt.xlabel('日期')
        plt.ylabel('AQI指数')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(monthly_data.index, monthly_data['PM2.5'], 'r-', label='PM2.5')
        plt.plot(monthly_data.index, monthly_data['PM10'], 'g-', label='PM10')
        plt.title('广州PM2.5和PM10月度趋势', fontsize=14)
        plt.xlabel('日期')
        plt.ylabel('浓度 (μg/m³)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig('monthly_trends.png', dpi=300)
        plt.close()
    except Exception as e:
        print(f"    创建月度趋势图时出错: {e}")
        print("    尝试替代方法...")

        # 替代方法：按年月进行分组
        trend_data['year'] = trend_data.index.year
        trend_data['month'] = trend_data.index.month
        alt_monthly = trend_data.groupby(['year', 'month'])[pollution_cols].mean().reset_index()

        # 创建日期列用于绘图
        alt_monthly['date'] = pd.to_datetime(
            alt_monthly['year'].astype(str) + '-' + alt_monthly['month'].astype(str) + '-01')
        alt_monthly.sort_values('date', inplace=True)

        plt.figure(figsize=(14, 8))
        plt.subplot(2, 1, 1)
        plt.plot(alt_monthly['date'], alt_monthly['AQI_index'], 'b-', label='AQI指数')
        plt.title('广州空气质量指数(AQI)月度趋势', fontsize=14)
        plt.xlabel('日期')
        plt.ylabel('AQI指数')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(alt_monthly['date'], alt_monthly['PM2.5'], 'r-', label='PM2.5')
        plt.plot(alt_monthly['date'], alt_monthly['PM10'], 'g-', label='PM10')
        plt.title('广州PM2.5和PM10月度趋势', fontsize=14)
        plt.xlabel('日期')
        plt.ylabel('浓度 (μg/m³)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        plt.savefig('monthly_trends.png', dpi=300)
        plt.close()

    # 3.3 污染物相关性分析
    print("  - 3.3 污染物相关性分析")

    # 确保所有列都是数值类型
    numeric_df = df[pollution_cols].copy()
    for col in pollution_cols:
        numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')

    # 计算相关性
    correlation = numeric_df.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f', linewidths=.5)
    plt.title('污染物相关性热力图', fontsize=14)
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300)
    plt.close()

    # 3.4 季节性分析
    print("  - 3.4 季节性分析")

    # 创建数值型数据副本
    numeric_df = df.copy()
    for col in pollution_cols:
        numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')

    # 确保季节是正确的字符串分类
    season_order = ['春季', '夏季', '秋季', '冬季']

    try:
        plt.figure(figsize=(14, 10))

        # 季节AQI对比
        plt.subplot(2, 2, 1)
        if 'season' in numeric_df.columns:
            sns.boxplot(x='season', y='AQI_index', data=numeric_df, order=season_order)
            plt.title('不同季节AQI指数分布', fontsize=12)
            plt.xlabel('季节')
            plt.ylabel('AQI指数')
        else:
            print("    警告: 'season'列不存在，跳过季节箱线图")

        # 月度PM2.5对比
        plt.subplot(2, 2, 2)
        if 'month' in numeric_df.columns:
            monthly_pm25 = numeric_df.groupby('month')['PM2.5'].mean().reset_index()
            sns.barplot(x='month', y='PM2.5', data=monthly_pm25)
            plt.title('月度PM2.5平均浓度', fontsize=12)
            plt.xlabel('月份')
            plt.ylabel('PM2.5 (μg/m³)')
        else:
            print("    警告: 'month'列不存在，跳过月度PM2.5图")

        # 工作日与周末对比
        plt.subplot(2, 2, 3)
        if 'is_weekend' in numeric_df.columns:
            # 确保is_weekend是数值类型
            numeric_df['is_weekend'] = pd.to_numeric(numeric_df['is_weekend'], errors='coerce')

            # 进行分组计算
            try:
                weekday_data = numeric_df.groupby('is_weekend')[pollution_cols].mean().reset_index()
                weekday_data = pd.melt(weekday_data, id_vars=['is_weekend'], value_vars=pollution_cols)
                weekday_data['is_weekend'] = weekday_data['is_weekend'].map({0: '工作日', 1: '周末'})
                sns.barplot(x='variable', y='value', hue='is_weekend', data=weekday_data)
                plt.title('工作日vs周末污染物平均值', fontsize=12)
                plt.xlabel('污染物')
                plt.ylabel('平均浓度')
                plt.xticks(rotation=45)
                plt.legend(title='')
            except Exception as e:
                print(f"    创建工作日vs周末对比图时出错: {e}")
        else:
            print("    警告: 'is_weekend'列不存在，跳过工作日vs周末对比图")

        # 年度变化趋势
        plt.subplot(2, 2, 4)
        if 'year' in numeric_df.columns:
            try:
                # 确保year是数值类型
                numeric_df['year'] = pd.to_numeric(numeric_df['year'], errors='coerce')

                yearly_data = numeric_df.groupby('year')[pollution_cols].mean().reset_index()
                yearly_data = pd.melt(yearly_data, id_vars=['year'], value_vars=['PM2.5', 'PM10', 'SO2', 'NO2'])
                sns.lineplot(x='year', y='value', hue='variable', data=yearly_data, marker='o')
                plt.title('主要污染物年度趋势', fontsize=12)
                plt.xlabel('年份')
                plt.ylabel('平均浓度')
                plt.grid(True, linestyle='--', alpha=0.5)
            except Exception as e:
                print(f"    创建年度变化趋势图时出错: {e}")
        else:
            print("    警告: 'year'列不存在，跳过年度变化趋势图")

        plt.tight_layout()
        plt.savefig('seasonal_analysis.png', dpi=300)
        plt.close()
    except Exception as e:
        print(f"    季节性分析时出错: {e}")

    # 3.5 AQI等级分布
    print("  - 3.5 AQI等级分布分析")

    # 定义AQI等级
    def aqi_category(aqi):
        if aqi <= 50:
            return '优'
        elif aqi <= 100:
            return '良'
        elif aqi <= 150:
            return '轻度污染'
        elif aqi <= 200:
            return '中度污染'
        elif aqi <= 300:
            return '重度污染'
        else:
            return '严重污染'

    df['aqi_category'] = df['AQI_index'].apply(aqi_category)

    # 绘制AQI等级饼图
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    aqi_counts = df['aqi_category'].value_counts()
    plt.pie(aqi_counts, labels=aqi_counts.index, autopct='%1.1f%%',
            colors=sns.color_palette('Set3', len(aqi_counts)),
            startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 1})
    plt.title('AQI等级分布', fontsize=14)

    # 按年份的AQI分布变化
    plt.subplot(1, 2, 2)
    yearly_aqi = df.groupby(['year', 'aqi_category']).size().unstack().fillna(0)
    yearly_aqi = yearly_aqi.div(yearly_aqi.sum(axis=1), axis=0) * 100  # 转为百分比

    yearly_aqi.plot(kind='bar', stacked=True, colormap='Set3')
    plt.title('各年AQI等级分布变化', fontsize=14)
    plt.xlabel('年份')
    plt.ylabel('百分比 (%)')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.legend(title='AQI等级', loc='upper right')

    plt.tight_layout()
    plt.savefig('aqi_distribution.png', dpi=300)
    plt.close()

    print("  - 探索性分析完成，图表已保存")


# 4. 时间序列分析
def time_series_analysis(df):
    """
    时间序列分析

    参数:
        df: 预处理后的数据
    """
    print("\n4. 时间序列分析...")

    # 确保数据按日期排序
    df = df.sort_values('date')

    # 4.1 季节性分解
    print("  - 4.1 AQI时间序列季节性分解")

    # 创建月度时间序列
    monthly_aqi = df.set_index('date')['AQI_index'].resample('M').mean()

    # 确保时间序列完整（无缺失）
    monthly_aqi = monthly_aqi.interpolate()

    # 进行季节性分解
    try:
        decomposition = seasonal_decompose(monthly_aqi, model='additive', period=12)

        plt.figure(figsize=(14, 10))
        plt.subplot(411)
        plt.plot(decomposition.observed)
        plt.title('观测值', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)

        plt.subplot(412)
        plt.plot(decomposition.trend)
        plt.title('趋势', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)

        plt.subplot(413)
        plt.plot(decomposition.seasonal)
        plt.title('季节性', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)

        plt.subplot(414)
        plt.plot(decomposition.resid)
        plt.title('残差', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig('seasonal_decomposition.png', dpi=300)
        plt.close()
    except Exception as e:
        print(f"    季节性分解失败: {e}")

    # 4.2 平稳性检验
    print("  - 4.2 平稳性检验 (ADF检验)")
    pollution_cols = ['AQI_index', 'PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']

    for col in pollution_cols:
        adf_result = adfuller(df[col].dropna())
        print(f"    {col} ADF检验:")
        print(f"      ADF统计量: {adf_result[0]:.4f}")
        print(f"      p值: {adf_result[1]:.4f}")
        print(f"      {'数据平稳' if adf_result[1] < 0.05 else '数据不平稳'}")

    # 4.3 自相关分析
    print("  - 4.3 AQI自相关分析")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))

    # 自相关函数
    sm.graphics.tsa.plot_acf(df['AQI_index'].dropna(), lags=30, ax=ax1)
    ax1.set_title('AQI指数自相关函数 (ACF)', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # 偏自相关函数
    sm.graphics.tsa.plot_pacf(df['AQI_index'].dropna(), lags=30, ax=ax2)
    ax2.set_title('AQI指数偏自相关函数 (PACF)', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('autocorrelation.png', dpi=300)
    plt.close()
def split_data(data, look_back=30):
    # 构造滑动窗口 X,y
    X, y = [], []
    for i in range(look_back, len(data)):
        X.append(data[i-look_back:i, :-1])
        y.append(data[i, -1])
    X, y = np.array(X), np.array(y)
    # 先按时间顺序划分 80% 训练，其余 20%
    n = len(X)
    idx = int(n * 0.8)
    X_train_full, y_train_full = X[:idx], y[:idx]
    X_temp, y_temp = X[idx:], y[idx:]
    # 再将剩余 20% 等分为 验证+测试 各 10%
    half = len(X_temp) // 2
    X_val, y_val = X_temp[:half], y_temp[:half]
    X_test, y_test = X_temp[half:], y_temp[half:]
    return X_train_full, y_train_full, X_val, y_val, X_test, y_test
def tune_lstm(train_data, look_back, coarse_params, fine_params):
    """
    基于“粗→细”网格搜索调优 LSTM 超参数。
    参数：
      - train_data: 同 build_lstm_model
      - look_back: 滑动窗口
      - coarse_params: 粗网格字典
      - fine_params: 细网格字典（只在粗最优参数附近再遍历）
    返回：
      字典，包含最优 model、scaler_X、scaler_y、history 及 params
    """
    best = {'mae': np.inf, 'params': None,
            'model': None, 'scaler_X': None, 'scaler_y': None, 'history': None}

    # —— 粗搜索 ——
    for p in ParameterGrid(coarse_params):
        m, sX, sy, h = build_lstm_model(
            train_data, look_back,
            the_lstm_layers = p.get('the_lstm_layers', 2),
            the_dense_layers= p.get('the_dense_layers',2),
            the_units       = p['the_units'],
            dropout_rate    = p['dropout_rate'],
            learning_rate   = p['learning_rate']
        )
        val_mae = h.history['val_loss'][-1]
        if val_mae < best['mae']:
            best.update({'mae': val_mae, 'params': p,
                         'model': m, 'scaler_X': sX, 'scaler_y': sy, 'history': h})

    # —— 细搜索 ——
    fine_grid = []
    for key, vs in fine_params.items():
        for v in vs:
            new = best['params'].copy()
            new[key] = v
            fine_grid.append(new)

    for p in fine_grid:
        m, sX, sy, h = build_lstm_model(
            train_data, look_back,
            the_lstm_layers = p.get('the_lstm_layers', 2),
            the_dense_layers= p.get('the_dense_layers',2),
            the_units       = p['the_units'],
            dropout_rate    = p['dropout_rate'],
            learning_rate   = p['learning_rate']
        )
        val_mae = h.history['val_loss'][-1]
        if val_mae < best['mae']:
            best.update({'mae': val_mae, 'params': p,
                         'model': m, 'scaler_X': sX, 'scaler_y': sy, 'history': h})

    print("LSTM 最优参数：", best['params'], "验证 MAE：", best['mae'])
    return best
def build_lstm_model(
    train_data,
    look_back=30,
    the_lstm_layers=2,
    the_dense_layers=2,
    the_units=32,
    dropout_rate=0.2,
    learning_rate=0.005
):
    """
    可传超参数的 LSTM 构建与训练函数。
    参数：
      - train_data: numpy 数组，形状 [n_samples, n_features+1]，最后一列为目标
      - look_back: 滑动窗口长度
      - the_lstm_layers: LSTM 层数
      - the_dense_layers: Dense 层数
      - the_units: 每层 LSTM/​Dense 的单元数
      - dropout_rate: Dropout 比例
      - learning_rate: Adam 学习率
    返回：
      model, scaler_X, scaler_y, history
    """
    # —— 数据预处理、归一化、滑动窗口 ——
    X_raw = train_data[:, :-1]
    y_raw = train_data[:, -1].reshape(-1,1)
    scaler_X = MinMaxScaler().fit(X_raw)
    scaler_y = MinMaxScaler().fit(y_raw)
    Xs = scaler_X.transform(X_raw)
    ys = scaler_y.transform(y_raw)

    X, y = [], []
    for i in range(look_back, len(Xs)):
        X.append(Xs[i-look_back:i])
        y.append(ys[i])
    X, y = np.array(X), np.array(y)

    # 划分训练/验证（90%/10%）
    split = int(len(X)*0.9)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    # —— 搭建网络 ——
    model = tf.keras.Sequential()
    # 第一层：双向 LSTM
    model.add(Bidirectional(LSTM(the_units, return_sequences=True),
                            input_shape=(look_back, X.shape[2])))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    # 中间 LSTM 层
    for _ in range(the_lstm_layers-1):
        model.add(LSTM(the_units, return_sequences=True,
                       kernel_regularizer=tf.keras.regularizers.l2(1e-4)))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
    # 最后一层 LSTM（不返回序列）
    model.add(LSTM(the_units, activation='tanh'))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))
    # 全连接层
    for _ in range(the_dense_layers):
        model.add(Dense(the_units, activation='relu',
                        kernel_regularizer=tf.keras.regularizers.l2(1e-4)))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))
    # 输出
    model.add(Dense(1, activation='linear'))

    # —— 编译 & 回调 ——
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss='mae',
        metrics=['mae']
    )
    cbs = [
        EarlyStopping('val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint('best_lstm.h5', save_best_only=True, monitor='val_loss'),
        ReduceLROnPlateau('val_loss', factor=0.5, patience=5)
    ]

    # —— 训练 ——
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=32,
        callbacks=cbs,
        verbose=2
    )

    return model, scaler_X, scaler_y, history

def tune_gru(train_data, look_back, coarse_params, fine_params):
    best = {'mae': np.inf, 'params': None, 'model':None, 'scaler':None, 'history':None}

    # —— 粗搜索 ——
    for p in ParameterGrid(coarse_params):
        m, s, h = build_gru_model(
            train_data, look_back,
            the_units     = p['the_units'],
            dense_layers  = p['dense_layers'],
            dropout_rate  = p['dropout_rate'],
            learning_rate = p['learning_rate']
        )
        mae = h.history['val_loss'][-1]
        if mae < best['mae']:
            best.update({'mae':mae, 'params':p, 'model':m, 'scaler':s, 'history':h})

    # —— 细搜索 ——
    fine_grid = []
    for key, vs in fine_params.items():
        for v in vs:
            new = best['params'].copy()
            new[key] = v
            fine_grid.append(new)
    for p in fine_grid:
        m, s, h = build_gru_model(
            train_data, look_back,
            the_units     = p['the_units'],
            dense_layers  = p['dense_layers'],
            dropout_rate  = p['dropout_rate'],
            learning_rate = p['learning_rate']
        )
        mae = h.history['val_loss'][-1]
        if mae < best['mae']:
            best.update({'mae':mae, 'params':p, 'model':m, 'scaler':s, 'history':h})

    print("GRU 最优参数：", best['params'], "验证 MAE：", best['mae'])
    return best

def build_gru_model(
    train_data,
    look_back=30,
    the_units=24,           # 新增：GRU 单元数可调
    dense_layers=[24,12],   # 新增：后续 Dense 层结构可调
    dropout_rate=0.1,       # 新增：Dropout 比例可调
    learning_rate=0.001     # 新增：学习率可调
):
    # 分离特征和目标（假设最后一列是目标变量）
    X_raw = train_data[:, :-1]  # 特征
    y_raw = train_data[:, -1]   # 目标

    # 仅归一化特征
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # 构造滑动窗口
    X, y = [], []
    for i in range(look_back, len(X_scaled)):
        X.append(X_scaled[i - look_back:i, :])
        y.append(y_raw[i])
    X, y = np.array(X), np.array(y)

    # 根据图片中的模型结构定义GRU模型
    model = tf.keras.Sequential()
    model.add(GRU(the_units, input_shape=(look_back, X.shape[2])))  # 用 the_units
    model.add(Dropout(dropout_rate))  # 用 dropout_rate
    for u in dense_layers:  # 用 dense_layers 列表
        model.add(Dense(u, activation='relu'))
        model.add(Dropout(dropout_rate))
    model.add(Dense(1))

    # 优化器和损失函数（与图片一致）
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=opt,
        loss='mae',
        metrics=['mae']
    )

    # 回调函数（早停、保存最优模型、学习率衰减）
    cbs = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint('best_gru.h5', save_best_only=True, monitor='val_loss'),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    ]

    # 训练模型
    history=model.fit(
        X, y,
        epochs=300,  # 与图片中的epoch数一致
        batch_size=64,
        validation_split=0.1,
        callbacks=cbs,
        verbose=2
    )
    return model, scaler,history


def predict_and_plot(df, lstm_model, gru_model, lstm_history, gru_history):
    look_back = 30
    features = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
    data = df[features + ['AQI_index']].values

    # —— 改动1：用 split_data 得到真正的 X_test, y_test ——
    _, _, X_val, y_val, X_test, y_test = split_data(data, look_back)

    # —— 改动2：构造与 X_test 对应的日期 ——
    all_dates    = pd.to_datetime(df['date'])
    window_dates = all_dates.iloc[look_back:].reset_index(drop=True)
    n    = len(window_dates)
    idx  = int(0.8 * n)
    half = (n - idx)//2
    test_dates = window_dates.iloc[idx+half : idx+half + len(y_test)].reset_index(drop=True)

    # 归一化（保持和训练时一致）
    feature_scaler = MinMaxScaler().fit(data[:, :-1])
    target_scaler  = MinMaxScaler().fit(data[:,  -1].reshape(-1,1))

    X_test_s = feature_scaler.transform(X_test.reshape(-1, len(features))).reshape(X_test.shape)

    # 预测
    y_pred_lstm = lstm_model.predict(X_test_s).flatten()
    y_pred_gru  = gru_model.predict(X_test_s).flatten()
    y_true = y_test
    y_pred_lstm = target_scaler.inverse_transform(y_pred_lstm.reshape(-1, 1)).flatten()
    #y_pred_gru = target_scaler.inverse_transform(y_pred_gru.reshape(-1, 1)).flatten()


    # ====== 新增评估指标计算 ======
    def calculate_ia(y_true, y_pred):
        """计算一致性指数(Index of Agreement)"""
        numerator = np.sum((y_pred - y_true) ** 2)
        denominator = np.sum((np.abs(y_pred - np.mean(y_true)) + np.abs(y_true - np.mean(y_true))) ** 2)
        return 1 - (numerator / denominator) if denominator != 0 else 0

    # LSTM评估指标
    mse_lstm = mean_squared_error(y_true, y_pred_lstm)
    rmse_lstm = np.sqrt(mse_lstm)
    mae_lstm = mean_absolute_error(y_true, y_pred_lstm)
    ia_lstm = calculate_ia(y_true, y_pred_lstm)

    # GRU评估指标
    mse_gru = mean_squared_error(y_true, y_pred_gru)
    rmse_gru = np.sqrt(mse_gru)
    mae_gru = mean_absolute_error(y_true, y_pred_gru)
    ia_gru = calculate_ia(y_true, y_pred_gru)


    norm_train = np.array(lstm_history.history['loss'])
    norm_val   = np.array(lstm_history.history['val_loss'])
    # 2. 计算放大因子：MinMaxScaler.scale_ = 1/(max-min)
    factor = 1.0 / target_scaler.scale_[0]
    # 3. 放大回原始 AQI 单位
    orig_train = norm_train * factor
    orig_val   = norm_val   * factor


    # 按年画图
    years = test_dates.dt.year.unique()
    # ====== 画 LSTM 图 ======
    plt.figure(figsize=(14, 7))
    for year in years:
        m = test_dates.dt.year == year
        plt.plot(test_dates[m], y_true[m],   'r-',  label=f'{year} True' if year==years[0] else "")
        plt.plot(test_dates[m], y_pred_lstm[m], 'b--', label=f'{year} LSTM Predict' if year==years[0] else "")
    plt.title('LSTM - Yearly AQI Prediction vs True Value')
    plt.xlabel('Date'); plt.ylabel('AQI')
    plt.legend(loc='upper left'); plt.xticks(rotation=45); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.show()

    # ====== 画 GRU 图 ======
    plt.figure(figsize=(14, 7))
    for year in years:
        m = test_dates.dt.year == year
        plt.plot(test_dates[m], y_true[m],  'r-',  label=f'{year} True' if year==years[0] else "")
        plt.plot(test_dates[m], y_pred_gru[m],  'g-.', label=f'{year} GRU Predict' if year==years[0] else "")
    plt.title('GRU - Yearly AQI Prediction vs True Value')
    plt.xlabel('Date'); plt.ylabel('AQI')
    plt.legend(loc='upper left'); plt.xticks(rotation=45); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.show()

    # —— 评价指标 ——
    # 输出评估结果
    print(f"\n—— 模型评价指标 ——")
    print(f"LSTM测试集:")
    print(f"  MSE: {mse_lstm:.4f}")
    print(f"  RMSE: {rmse_lstm:.4f}")
    print(f"  MAE: {mae_lstm:.4f}")
    print(f"  IA: {ia_lstm:.4f}")
    print(f"  R²: {r2_score(y_true, y_pred_lstm):.4f}")

    print(f"\nGRU测试集:")
    print(f"  MSE: {mse_gru:.4f}")
    print(f"  RMSE: {rmse_gru:.4f}")
    print(f"  MAE: {mae_gru:.4f}")
    print(f"  IA: {ia_gru:.4f}")
    print(f"  R²: {r2_score(y_true, y_pred_gru):.4f}")
    print(f"\n测试集样本数: {len(y_test)}")

    # ====== 画训练Loss曲线（新增部分）======


    # LSTM 曲线
    plt.figure(figsize=(8, 4))
    plt.plot(orig_train, label='LSTM 训练损失')
    plt.plot(orig_val, label='LSTM 验证损失')
    plt.title('LSTM Loss vs Val_Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    # GRU 曲线
    plt.figure(figsize=(8, 4))
    plt.plot(gru_history.history['loss'], label='GRU 训练损失')
    plt.plot(gru_history.history['val_loss'], label='GRU 验证损失')
    plt.title('GRU Loss vs Val_Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def main():
    print("============== 广州空气质量数据分析 ==============")
    file_path = "data/datas_guangzhou.csv"  # 修改为实际路径

    try:
        df = load_data(file_path)
        df_processed = preprocess_data(df)
        exploratory_data_analysis(df_processed)
        time_series_analysis(df_processed)
        predict_and_plot(df_processed)  # 新增预测流程
        print("\n分析完成！所有图表已保存。")
    except Exception as e:
        print(f"错误: {e}")
# 运行模型训练与预测
if __name__ == "__main__":
    # 1. 加载并预处理
    df = load_data(r"data\datas_guangzhou.csv")
    df = preprocess_data(df)
    features = ['PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
    data = df[features + ['AQI_index']].values

    # 2. 划分窗口只是为了得到 X_test, y_test；训练由 build_* 接管
    look_back = 30
    X_train_full, y_train_full, X_val, y_val, X_test, y_test = split_data(data, look_back)

    # 3. 直接使用原始 data 作为 train_data
    train_data = data

    # 构建并训练
    # LSTM 超参数调优 ——
    coarse_lstm = {
        'the_lstm_layers': [1, 2],
        'the_dense_layers': [1, 2],
        'the_units': [16, 32],
        'dropout_rate': [0.1, 0.2],
        'learning_rate': [0.01, 0.005]
    }
    fine_lstm = {
        'the_units': [24, 32],
        'learning_rate': [0.007, 0.003]
    }
    best_lstm = tune_lstm(
        train_data=data,
        look_back=look_back,
        coarse_params=coarse_lstm,
        fine_params=fine_lstm
    )
    # 最优 LSTM
    lstm_model, scaler_X, scaler_y, lstm_history = (
        best_lstm['model'],
        best_lstm['scaler_X'],
        best_lstm['scaler_y'],
        best_lstm['history']
    )
    # GRU 超参数调优 ——
    coarse_gru = {
        'the_units': [16, 24, 32],
        'dense_layers': [[16, 8], [24, 12]],
        'dropout_rate': [0.1, 0.2],
        'learning_rate': [0.01, 0.001]
    }
    fine_gru = {
        'the_units': [20, 24, 28],
        'dropout_rate': [0.05, 0.1],
        'learning_rate': [0.005, 0.002]
    }
    best_gru = tune_gru(
        train_data=data,
        look_back=look_back,
        coarse_params=coarse_gru,
        fine_params=fine_gru
    )
    # 最优 GRU
    gru_model, gru_scaler, gru_history = (
        best_gru['model'],
        best_gru['scaler'],
        best_gru['history']
    )

    predict_and_plot(df, lstm_model, gru_model, lstm_history, gru_history)

