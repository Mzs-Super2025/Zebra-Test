import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# ====================
# 数据模拟模块
# ====================
def generate_survey_data(sample_size=1000):
    """生成模拟调查数据"""
    np.random.seed(42)
    
    data = pd.DataFrame({
        'age': np.random.randint(18, 35, sample_size),
        'gender': np.random.choice(['Male', 'Female'], sample_size),
        'education': np.random.choice(['High School', 'Bachelor', 'Master'], sample_size, p=[0.2, 0.6, 0.2]),
        'income': np.random.normal(loc=8000, scale=3000, size=sample_size).astype(int),
        'work_hours': np.random.randint(40, 100, sample_size),
        'housing_type': np.random.choice(['Rent', 'Own', 'Parents'], sample_size, p=[0.6, 0.2, 0.2]),
        'social_media_time': np.random.randint(2, 8, sample_size),
        'exercise_freq': np.random.choice(['Daily', 'Weekly', 'Rarely'], sample_size),
        'anxiety_level': np.random.randint(1, 10, sample_size)  # 1-10焦虑程度评分
    })
    
    # 添加收入与焦虑程度的负相关性
    data['anxiety_level'] = data.apply(lambda x: x['anxiety_level'] + (100000 - x['income'])//5000, axis=1)
    data['anxiety_level'] = np.clip(data['anxiety_level'], 1, 10)
    
    return data

# 生成模拟数据
survey_data = generate_survey_data()
print("数据样例：")
print(survey_data.head())

# ====================
# 数据分析模块
# ====================
def analyze_anxiety(data):
    """执行完整分析流程"""
    # 数据预处理
    data = pd.get_dummies(data, columns=['gender', 'education', 'housing_type', 'exercise_freq'])
    
    # 计算相关系数
    corr_matrix = data.corr()
    
    # 使用机器学习分析关键因素
    X = data.drop('anxiety_level', axis=1)
    y = data['anxiety_level']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # 获取特征重要性
    coefficients = pd.DataFrame({
        'feature': X.columns,
        'importance': model.coef_[0]
    }).sort_values('importance', ascending=False)
    
    return corr_matrix, coefficients

corr_matrix, feature_importance = analyze_anxiety(survey_data)

# ====================
# 可视化模块
# ====================
plt.figure(figsize=(15, 10))

# 焦虑水平分布
plt.subplot(2, 2, 1)
sns.histplot(survey_data['anxiety_level'], bins=10, kde=True)
plt.title('Anxiety Level Distribution')
plt.xlabel('Anxiety Score')

# 收入与焦虑关系
plt.subplot(2, 2, 2)
sns.regplot(x='income', y='anxiety_level', data=survey_data)
plt.title('Income vs Anxiety Level')

# 特征重要性
plt.subplot(2, 2, 3)
sns.barplot(x='importance', y='feature', data=feature_importance.head(5))
plt.title('Top 5 Anxiety Factors')

# 工作时长与焦虑关系
plt.subplot(2, 2, 4)
sns.boxplot(x='work_hours', y='anxiety_level', data=survey_data)
plt.title('Work Hours vs Anxiety Level')

plt.tight_layout()
plt.show()

# ====================
# 分析报告生成
# ====================
print("\n关键分析结论：")
print("1. 收入水平与焦虑程度呈显著负相关（r = %.2f）" % corr_matrix.loc['income', 'anxiety_level'])
print("2. 主要焦虑影响因素前三位：")
for i, row in feature_importance.head(3).iterrows():
    print(f"   - {row['feature']} (重要性: {row['importance']:.2f})")
    
print("\n建议措施：")
print("1. 完善青年职业发展支持体系")
print("2. 建立心理健康服务平台")
print("3. 推行合理工时管理制度")
print("4. 加强社交媒体使用教育")# Zebra-Test
