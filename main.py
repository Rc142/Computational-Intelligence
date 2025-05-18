# 导入所需库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime
import warnings
from sklearn.ensemble import RandomForestRegressor
import matplotlib.dates as mdates
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from scipy.stats import randint, uniform

import matplotlib
matplotlib.use('Agg')


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
    print(f"加载数据文件: {file_path}")
    # 加载CSV文件
    df = pd.read_csv(file_path)
    print(f"  - 数据加载完成，共 {df.shape[0]} 行，{df.shape[1]} 列")
    return df


# 2. 数据预处理
def preprocess_data(df, is_validation=False):
    """
    数据预处理：清洗、转换和特征工程

    参数:
        df: 原始数据DataFrame
        is_validation: 是否是验证集数据
    返回:
        DataFrame: 预处理后的数据
    """
    print("\n数据预处理...")

    # 2.1 数据类型转换
    print("  - 转换日期格式")
    df['date'] = pd.to_datetime(df['date'])

    # 2.2 检查并处理缺失值
    print("  - 处理缺失值")
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print(f"    缺失值统计:\n{missing_values}")
        # 使用前后值插值填充缺失值
        df = df.interpolate(method='time')
        # 如果首尾有缺失值无法插值，使用最近的有效值填充
        df = df.fillna(method='bfill').fillna(method='ffill')
    else:
        print("    无缺失值")

    # 2.3 异常值检测与处理
    print("  - 检测和处理异常值")
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
    print("  - 特征工程")

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

    # 创建季节性sin-cos特征（对周期性建模有帮助）
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

    # 对于训练数据，创建滞后特征
    # 对于验证数据，滞后特征将在预测过程中处理
    if not is_validation:
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
        df_with_na = df.copy()
        df = df.dropna()
        print(f"  - 删除缺失值后数据形状: {df.shape} (删除了 {len(df_with_na) - len(df)} 行)")

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


# 4. 预测模型构建
def build_prediction_models(df):
    """
    构建和评估预测模型

    参数:
        df: 预处理后的数据
    返回:
        tuple: 包含模型性能评估结果和训练好的模型
    """
    print("\n4. 预测模型构建...")

    # 定义特征和目标变量
    target_col = 'AQI_index'  # 以AQI指数为预测目标

    # 排除非特征列与非数值列
    exclude_cols = ['date', 'aqi_category', target_col]

    # 识别分类特征
    categorical_cols = []
    numerical_cols = []

    for col in df.columns:
        if col not in exclude_cols and col != target_col:
            if col == 'season':  # 明确识别季节作为分类特征
                categorical_cols.append(col)
            elif df[col].dtype == 'object' or df[col].dtype == 'category':
                categorical_cols.append(col)
            else:
                numerical_cols.append(col)

    print(f"    分类特征: {categorical_cols}")
    print(f"    数值特征数量: {len(numerical_cols)}")

    # 准备特征和目标变量
    feature_cols = numerical_cols + categorical_cols

    # 基础机器学习模型的数据准备
    X = df[feature_cols].copy()
    y = pd.to_numeric(df[target_col], errors='coerce')

    # 处理可能的缺失值
    X = X.fillna(method='ffill').fillna(method='bfill')
    y = y.fillna(y.mean())

    # 创建列转换器来处理分类和数值特征
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', SimpleImputer(strategy='mean'), numerical_cols),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_cols)
        ],
        remainder='passthrough'
    )
    # 时间序列交叉验证设定
    tscv = TimeSeriesSplit(n_splits=5)
    # 划分训练集和测试集 (80% 训练, 20% 测试)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False)

    print(f"  - 划分数据集: 训练集 {X_train.shape[0]} 样本, 测试集 {X_test.shape[0]} 样本")

    # 定义评估指标函数
    def evaluate_model(y_true, y_pred, model_name):
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        print(f"    {model_name} 性能评估:")
        print(f"      RMSE: {rmse:.2f}")
        print(f"      MAE: {mae:.2f}")
        print(f"      R²: {r2:.4f}")

        return {
            'model': model_name,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }

    model_results = []
    trained_models = {}

    print("  - 4.1 机器学习模型训练与评估（随机搜索超参）")

    # 随机森林 + 随机搜索
    print("    训练随机森林模型 + 随机搜索...")
    rf_pipeline = Pipeline([('preprocessor', preprocessor),
                            ('model', RandomForestRegressor(random_state=42))])
    rf_param_dist = {
        'model__n_estimators': randint(50, 300),
        'model__max_depth': [None, 10, 20, 30],
        'model__min_samples_split': randint(2, 20)
    }
    rf_search = RandomizedSearchCV(
        rf_pipeline, param_distributions=rf_param_dist,
        n_iter=20, cv=tscv, scoring='neg_mean_squared_error',
        n_jobs=-1, random_state=42
    )
    rf_search.fit(X_train, y_train)
    rf_best = rf_search.best_estimator_
    print("      → 最优参数:", rf_search.best_params_)
    rf_pred = rf_best.predict(X_test)
    model_results.append(evaluate_model(y_test, rf_pred, "随机森林"))
    trained_models["随机森林"] = rf_best

    # XGBoost + 随机搜索
    print("    训练XGBoost模型 + 随机搜索...")
    xgb_pipeline = Pipeline([('preprocessor', preprocessor),
                             ('model', XGBRegressor(random_state=42, verbosity=0))])
    xgb_param_dist = {
        'model__n_estimators': randint(50, 300),
        'model__learning_rate': uniform(0.01, 0.3),
        'model__max_depth': [None, 10, 20, 30]
    }
    xgb_search = RandomizedSearchCV(
        xgb_pipeline, param_distributions=xgb_param_dist,
        n_iter=20, cv=tscv, scoring='neg_mean_squared_error',
        n_jobs=-1, random_state=42
    )
    xgb_search.fit(X_train, y_train)
    xgb_best = xgb_search.best_estimator_
    print("      → 最优参数:", xgb_search.best_params_)
    xgb_pred = xgb_best.predict(X_test)
    model_results.append(evaluate_model(y_test, xgb_pred, "XGBoost"))
    trained_models["XGBoost"] = xgb_best

    # SVR + 随机搜索
    print("    训练SVR模型 + 随机搜索...")
    svr_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('scaler', MinMaxScaler()),
        ('model', SVR())
    ])
    svr_param_dist = {
        'model__C': uniform(0.1, 100),
        'model__gamma': ['scale', 'auto'] + list(uniform(0.001, 1).rvs(3)),
        'model__epsilon': uniform(0.01, 1)
    }
    svr_search = RandomizedSearchCV(
        svr_pipeline, param_distributions=svr_param_dist,
        n_iter=20, cv=tscv, scoring='neg_mean_squared_error',
        n_jobs=-1, random_state=42
    )
    svr_search.fit(X_train, y_train)
    svr_best = svr_search.best_estimator_
    print("      → 最优参数:", svr_search.best_params_)
    svr_pred = svr_best.predict(X_test)
    model_results.append(evaluate_model(y_test, svr_pred, "SVR"))
    trained_models["SVR"] = svr_best

    # 线性回归（基线）
    print("    训练线性回归模型...")
    lr_pipeline = Pipeline([('preprocessor', preprocessor),
                            ('model', LinearRegression())])
    lr_pipeline.fit(X_train, y_train)
    lr_pred = lr_pipeline.predict(X_test)
    model_results.append(evaluate_model(y_test, lr_pred, "线性回归"))
    trained_models["线性回归"] = lr_pipeline

    # 4.2 特征重要性分析
    print("  - 4.2 特征重要性分析")

    # 提取随机森林的特征重要性
    try:
        # 使用独热编码时，特征名称会变化，需要获取转换后的特征名
        feature_names = numerical_cols.copy()
        for col in categorical_cols:
            # 获取类别数量
            unique_values = df[col].nunique()
            # OneHotEncoder的drop='first'选项会删除一个类别
            for i in range(unique_values - 1):
                feature_names.append(f"{col}_{i}")

        # 提取随机森林模型
        rf_model = trained_models["随机森林"].named_steps['model']
        # 获取特征重要性
        if len(feature_names) == len(rf_model.feature_importances_):
            importances = rf_model.feature_importances_
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)

            # 展示前15个最重要的特征
            top_n = min(15, len(feature_importance))
            top_features = feature_importance.head(top_n)

            plt.figure(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=top_features)
            plt.title('随机森林特征重要性 (Top 15)', fontsize=14)
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=300)
            plt.close()

            print(f"    Top {top_n} 重要特征:")
            print(top_features)
        else:
            print("    警告: 特征名称数量与特征重要性数量不匹配，无法生成特征重要性图")
    except Exception as e:
        print(f"    生成特征重要性分析时出错: {e}")
    # 获取XGBoost模型的特征重要性
    try:
        # 使用独热编码时，特征名称会变化，需要获取转换后的特征名
        feature_names = numerical_cols.copy()
        for col in categorical_cols:
            # 获取类别数量
            unique_values = df[col].nunique()
            # OneHotEncoder的drop='first'选项会删除一个类别
            for i in range(unique_values - 1):
                feature_names.append(f"{col}_{i}")

        # 提取XGBoost模型
        rf_model = trained_models["XGBoost"].named_steps['model']
        # 获取特征重要性
        if len(feature_names) == len(rf_model.feature_importances_):
            importances = rf_model.feature_importances_
            feature_importance = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)

            # 展示前15个最重要的特征
            top_n = min(15, len(feature_importance))
            top_features = feature_importance.head(top_n)

            plt.figure(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=top_features)
            plt.title('XGBoost特征重要性 (Top 15)', fontsize=14)
            plt.tight_layout()
            plt.savefig('XGBoostfeature_importance.png', dpi=300)
            plt.close()

            print(f"    Top {top_n} 重要特征:")
            print(top_features)
        else:
            print("    警告: 特征名称数量与特征重要性数量不匹配，无法生成特征重要性图")
    except Exception as e:
        print(f"    生成特征重要性分析时出错: {e}")


    # 获取线性回归模型的回归系数
    lr_model = trained_models["线性回归"].named_steps['model']
    importances_lr = lr_model.coef_

    # 生成特征重要性数据
    feature_importance_lr = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances_lr
    }).sort_values('Importance', ascending=False)

    # 展示前15个最重要的特征
    top_features_lr = feature_importance_lr.head(15)

    # 可视化
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=top_features_lr)
    plt.title('线性回归特征重要性 (Top 15)', fontsize=14)
    plt.tight_layout()
    plt.savefig('lr_feature_importance.png', dpi=300)
    plt.close()

    # 获取线性回归模型
    lr_model = trained_models["线性回归"].named_steps['model']

    # 打印回归系数（即权重）
    coefficients = lr_model.coef_

    # 将特征名称与对应的回归系数结合起来，形成一个数据框，便于查看
    coeff_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    }).sort_values('Coefficient', ascending=False)

    # 打印回归系数
    print("线性回归模型的回归系数（权重）:")
    print(coeff_df)

    # 4.3 模型对比
    print("  - 4.3 模型性能对比")

    # 将结果转换为DataFrame便于比较
    results_df = pd.DataFrame(model_results)

    # 绘制模型性能对比
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    bars = plt.bar(results_df['model'], results_df['rmse'])
    plt.title('各模型RMSE对比（越低越好）', fontsize=14)
    plt.ylabel('RMSE')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                 f'{height:.2f}', ha='center', va='bottom')

    plt.subplot(2, 1, 2)
    bars = plt.bar(results_df['model'], results_df['r2'])
    plt.title('各模型$R^2$对比（越高越好）', fontsize=14)
    plt.ylabel('R²')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300)
    plt.close()

    # 4.4 最佳模型预测可视化
    print("  - 4.4 最佳模型预测可视化")

    # 找出最佳模型（基于R²）
    best_model_idx = results_df['r2'].idxmax()
    best_model = results_df.loc[best_model_idx, 'model']

    # 获取对应的预测结果
    if best_model == "随机森林":
        best_pred = rf_pred
    elif best_model == "XGBoost":
        best_pred = xgb_pred
    elif best_model == "SVR":
        best_pred = svr_pred
    else:
        best_pred = lr_pred

    # 获取测试集日期
    test_dates = df.iloc[-len(y_test):]['date'].values

    # 绘制实际值与预测值对比
    plt.figure(figsize=(14, 8))

    plt.plot(test_dates, y_test, 'b-', label='实际值', linewidth=2)
    plt.plot(test_dates, best_pred, 'r-', label=f'{best_model}预测值', linewidth=2)

    plt.title(f'AQI指数: 实际值 vs {best_model}预测值', fontsize=16)
    plt.xlabel('日期')
    plt.ylabel('AQI指数')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # 格式化x轴日期
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.gcf().autofmt_xdate()

    plt.tight_layout()
    plt.savefig('best_model_prediction.png', dpi=300)
    plt.close()

    print(f"  - 最佳模型是 {best_model}，预测可视化已完成")

    return results_df, trained_models


# 5. 验证集评估
def evaluate_on_validation_set(models, training_data, validation_data, pollution_cols):
    """
    在2025年1-4月验证集上评估模型性能

    参数:
        models: 训练好的模型字典
        training_data: 训练数据
        validation_data: 验证数据
        pollution_cols: 污染物列名列表
    """
    print("\n5. 验证集评估...")

    # 准备验证集特征
    val_features = validation_data.copy()

    # 为验证数据添加滞后特征
    # 由于验证数据需要使用训练数据的最后几条记录来计算滞后特征
    # 创建一个合并数据集，包含训练数据的最后一个月和验证数据
    print("  - 准备验证数据的滞后特征")

    # 获取训练数据的最后30天
    last_date = training_data['date'].max()
    thirty_days_before = last_date - pd.Timedelta(days=30)
    training_tail = training_data[training_data['date'] > thirty_days_before].copy()

    # 合并数据
    combined_data = pd.concat([training_tail, val_features], ignore_index=True)
    combined_data = combined_data.sort_values('date')

    # 创建滞后特征
    for col in pollution_cols:
        # 前1天的值
        combined_data[f'{col}_lag1'] = combined_data[col].shift(1)
        # 前7天的值
        combined_data[f'{col}_lag7'] = combined_data[col].shift(7)
        # 前7天的平均值
        combined_data[f'{col}_rolling7'] = combined_data[col].rolling(window=7).mean()

    # 分离出验证数据
    val_features = combined_data[combined_data['date'] >= validation_data['date'].min()].copy()

    # 确保没有缺失值
    val_features = val_features.fillna(method='ffill').fillna(method='bfill')

    # 记录真实的AQI值用于评估
    y_true = val_features['AQI_index'].values

    # 排除非特征列
    exclude_cols = ['date', 'aqi_category', 'AQI_index']
    feature_cols = [col for col in val_features.columns if col not in exclude_cols]
    X_val = val_features[feature_cols]

    # 评估各个模型
    val_results = []
    all_predictions = {}

    for model_name, model in models.items():
        print(f"  - 评估模型: {model_name}")

        # 预测验证集
        y_pred = model.predict(X_val)

        # 保存预测结果
        all_predictions[model_name] = y_pred

        # 计算评估指标
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        print(f"    RMSE: {rmse:.2f}")
        print(f"    MAE: {mae:.2f}")
        print(f"    R²: {r2:.4f}")

        val_results.append({
            'model': model_name,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        })

    # 将结果转换为DataFrame
    val_results_df = pd.DataFrame(val_results)

    # 绘制验证集性能对比
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    bars = plt.bar(val_results_df['model'], val_results_df['rmse'])
    plt.title('各模型在2025年验证集上的RMSE对比（越低越好）', fontsize=14)
    plt.ylabel('RMSE')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                 f'{height:.2f}', ha='center', va='bottom')

    plt.subplot(2, 1, 2)
    bars = plt.bar(val_results_df['model'], val_results_df['r2'])
    plt.title('各模型在2025年验证集上的R²对比（越高越好）', fontsize=14)
    plt.ylabel('R²')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                 f'{height:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig('validation_performance.png', dpi=300)
    plt.close()

    # 绘制验证集上的预测结果对比
    plt.figure(figsize=(14, 10))

    # 绘制所有模型预测结果
    dates = val_features['date']
    plt.plot(dates, y_true, 'k-', label='实际值', linewidth=2)

    colors = ['r-', 'g-', 'b-', 'm-']
    for i, (model_name, predictions) in enumerate(all_predictions.items()):
        plt.plot(dates, predictions, colors[i % len(colors)], label=f'{model_name}预测值', linewidth=1.5, alpha=0.8)

    plt.title('2025年1-4月验证集上各模型预测结果对比', fontsize=16)
    plt.xlabel('日期')
    plt.ylabel('AQI指数')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    # 格式化x轴日期
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=10))
    plt.gcf().autofmt_xdate()

    plt.tight_layout()
    plt.savefig('validation_predictions.png', dpi=300)
    plt.close()

    # 找出验证集上最佳模型
    best_model_idx = val_results_df['r2'].idxmax()
    best_model = val_results_df.loc[best_model_idx, 'model']
    best_r2 = val_results_df.loc[best_model_idx, 'r2']

    print(f"\n验证集上的最佳模型: {best_model}，R² = {best_r2:.4f}")

    # 返回验证结果
    return val_results_df, all_predictions


# 主函数
def main():
    """
    主函数：调用所有分析步骤
    """
    print("============== 广州空气质量数据分析 ==============")

    # 1. 加载训练数据
    train_file_path = r"data\datas_guangzhou.csv"
    df_train = load_data(train_file_path)

    # 2. 加载验证数据
    validation_file_path = r"data\guangzhou_2025_air_quality.csv"
    df_validation = load_data(validation_file_path)

    # 3. 数据预处理
    df_train_processed = preprocess_data(df_train)
    df_validation_processed = preprocess_data(df_validation, is_validation=True)

    # 确保主要污染物列都存在于验证集中
    pollution_cols = ['AQI_index', 'PM2.5', 'PM10', 'SO2', 'NO2', 'CO', 'O3']
    for col in pollution_cols:
        if col not in df_validation_processed.columns:
            print(f"警告: 验证集中缺少 '{col}' 列")

    # 4. 探索性数据分析（仅对训练数据）
    exploratory_data_analysis(df_train_processed)

    # 5. 预测模型构建
    model_results, trained_models = build_prediction_models(df_train_processed)

    # 6. 在验证集上评估模型
    validation_results, validation_predictions = evaluate_on_validation_set(
        trained_models, df_train_processed, df_validation_processed, pollution_cols)

    # 7. 输出总结信息
    print("\n分析完成！所有图表已保存。")

    print(
        f"训练数据: {len(df_train)} 条记录，跨度从 {df_train['date'].min().strftime('%Y-%m-%d')} 到 {df_train['date'].max().strftime('%Y-%m-%d')}")
    print(
        f"验证数据: {len(df_validation)} 条记录，跨度从 {df_validation['date'].min().strftime('%Y-%m-%d')} 到 {df_validation['date'].max().strftime('%Y-%m-%d')}")

    # 训练集上的最佳模型
    train_best_model = model_results.loc[model_results['r2'].idxmax(), 'model']
    train_best_r2 = model_results.loc[model_results['r2'].idxmax(), 'r2']
    print(f"\n训练集上的最佳模型: {train_best_model}，R² = {train_best_r2:.4f}")

    # 验证集上的最佳模型
    val_best_model = validation_results.loc[validation_results['r2'].idxmax(), 'model']
    val_best_r2 = validation_results.loc[validation_results['r2'].idxmax(), 'r2']
    print(f"验证集上的最佳模型: {val_best_model}，R² = {val_best_r2:.4f}")


if __name__ == "__main__":
    main()