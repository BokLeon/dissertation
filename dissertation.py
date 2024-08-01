#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sb
import matplotlib.pyplot as plt
from wordcloud import WordCloud, ImageColorGenerator
from matplotlib import rcParams
rcParams['figure.figsize'] = (15, 8)

#ignoring all the warnings because we don't need them
import warnings
warnings.filterwarnings('ignore')

# 读取 CSV 文件到 DataFrame 中
df = pd.read_csv('police_stop_data.csv')

print(df.head())

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[3]:


stop_data = pd.read_csv('police_stop_data.csv')
force_data = pd.read_csv('police_use_of_force.csv')
stop_data.head(3)


# In[4]:


stop_data.shape


# In[5]:


stop_data.describe()


# In[6]:


print("Missing Data")
miss_me = stop_data.isnull().sum()
miss_me[miss_me>0]


# In[7]:


categorical_cols = [col for col in stop_data.columns
                    if stop_data[col].dtype=='object']

print(f"Categorical Columns:\n{categorical_cols}")


# In[8]:


census_2010 = {'white': ['63.8'], 
        'black': ['18.6'], 
        'hispanic': ['10.5'],
        'asian': ['5.6'],
        'other': ['5.6'],
        'american_indian': ['2.0']}

census_2010_df = pd.DataFrame(census_2010, columns = ['white','black','hispanic','asian','other','american_indian']).transpose()
census_2010_df.columns = ['Percentage of Population']
census_2010_df['Percentage of Population'] = census_2010_df['Percentage of Population'].astype(float)


# In[9]:


census_2010_df.style.background_gradient(cmap='Purples', subset=['Percentage of Population'])


# In[10]:


# Bar Plot
fig1 = px.bar(census_2010_df, x=census_2010_df.index, 
              y=census_2010_df['Percentage of Population'], color=census_2010_df.index, 
              color_discrete_sequence=px.colors.qualitative.Pastel)

fig1.update_layout(title={
                  'text': "Demographics of Minneapolis City",
                  'y':0.98,
                  'x':0.5,
                  'xanchor': 'center',
                  'yanchor': 'top'},
                  template='ggplot2')


# In[11]:


# Pie Chart
fig2 = px.pie(census_2010_df, census_2010_df.index, 
              census_2010_df['Percentage of Population'], 
              color_discrete_sequence=px.colors.qualitative.Pastel, hole=0.5)

fig2.update_layout(title={
                  'text': "Demographics of Minneapolis City (Pie Chart)",
                  'y':0.98,
                  'x':0.5,
                  'xanchor': 'center',
                  'yanchor': 'top'},
                   height=600,
                  template='plotly_white')

fig2.update_traces(textposition='inside', textinfo='percent+label', pull=[0, 0.2])

fig2.data[0].marker.line.width = 1
fig2.data[0].marker.line.color = "black"

fig2.show()


# In[11]:


stop_data['race'].value_counts()


# In[12]:


# Bar Plot
fig1 = px.bar(stop_data['race'].value_counts(), color_discrete_sequence=[px.colors.qualitative.Pastel])

fig1.update_layout(title={
                  'text': "Minneapolis Police Stops by Race",
                  'y':0.98,
                  'x':0.5,
                  'xanchor': 'center',
                  'yanchor': 'top'}, 
                  template='ggplot2', 
                  xaxis_title="Race", 
                  yaxis_title="Total Number",
                  showlegend=False
                 )

# -------------------------------------------------------------

# Funnel Plot
fig2 = px.funnel(stop_data['race'].value_counts(), color_discrete_sequence=[px.colors.qualitative.Pastel])
fig2.update_layout(template='ggplot2', showlegend=False)

fig1.show()
fig2.show()


# In[13]:


# Fill missing values with 'None'
stop_data['reason'] = stop_data['reason'].fillna('None')

stop_reason = stop_data['reason'].unique()

print("Different Reasons for Stopping the vehicle:")
for x in stop_reason:
    print(x)


# In[14]:


print("Total Number of Cases:")
stop_data['reason'].value_counts()


# In[13]:


# Bar Chart
fig1 = px.bar(stop_data['reason'].value_counts(), color_discrete_sequence=[px.colors.qualitative.Pastel])

fig1.update_layout(title={
                  'text': "Reason for Police Stops",
                  'y':0.98,
                  'x':0.5,
                  'xanchor': 'center',
                  'yanchor': 'top'}, 
                  template='ggplot2', 
                  xaxis_title="Reason", 
                  yaxis_title="Total Number",
                  showlegend=False
                 )


# In[14]:


# Pie Chart
fig2 = px.pie(stop_data, stop_data['reason'], 
              color_discrete_sequence=px.colors.qualitative.Pastel, hole=0.3)

fig2.update_layout(title={
                  'text': "Reasons for Police Stops (Pie Chart)",
                  'y':0.98,
                  'x':0.5,
                  'xanchor': 'center',
                  'yanchor': 'top'},
                   height=600,
                  template='plotly_white')

fig2.update_traces(textposition='inside', textinfo='percent+label', pull=(0.05))

fig2.data[0].marker.line.width = 1
fig2.data[0].marker.line.color = "black"
fig2.show()


# In[15]:


fig = px.bar(stop_data['problem'].value_counts(), color_discrete_sequence=[px.colors.qualitative.Pastel])

fig.update_layout(title={
                  'text': "Problem Stated by Police",
                  'y':0.98,
                  'x':0.5,
                  'xanchor': 'center',
                  'yanchor': 'top'}, 
                  template='ggplot2', 
                  xaxis_title="Problem", 
                  yaxis_title="Total Number",
                  showlegend=False
                 )

fig.show()


# In[17]:


problem_stop = stop_data.groupby('race')[['problem']].count().reset_index()
problem_stop.sort_values(by='problem',  ascending=False).style.background_gradient(cmap='Reds', subset=['problem'])


# In[18]:


fig1 = px.bar(problem_stop, problem_stop['race'], 
              problem_stop['problem'], 
              color_discrete_sequence = [px.colors.qualitative.Plotly])

fig1.update_layout(title={
                  'text': "Problem based on Race",
                  'y':0.98,
                  'x':0.5,
                  'xanchor': 'center',
                  'yanchor': 'top'}, 
                  template='ggplot2', 
                  xaxis_title="Race", 
                  yaxis_title="Problem",
                 )

# ----------------------------------------------------

fig2 = go.Figure(data=[go.Scatter(
    x=problem_stop['race'], y=problem_stop['problem'],
    mode='markers',
    marker=dict(
        color= px.colors.qualitative.Plotly,
        size=[20, 140, 60, 50, 40, 30, 70, 90],
    )
    )])

fig2.update_layout(template='ggplot2',
                   xaxis_title='Race',
                   yaxis_title='Problem')

fig1.show()
fig2.show()


# In[20]:


word = stop_data['callDisposition']

text = " ".join(str(each) for each in word.unique())

wordcloud = WordCloud(max_words=200, colormap='Set3', background_color="white").generate(text)

plt.figure(figsize=(17,10))

plt.imshow(wordcloud, interpolation='Bilinear')

plt.axis("off")

plt.figure(1,figsize=(12, 12))

plt.title("Word Cloud of Call Disposition", size=20)

plt.show()


# In[19]:


personSearch_df = stop_data[stop_data['personSearch'] == 'YES']

search_person = personSearch_df.groupby('race')['personSearch'].count().reset_index()

search_person.sort_values(by='personSearch', ascending=False).style.background_gradient(cmap='Oranges', subset=['personSearch'])


# In[20]:


fig = px.pie(search_person, search_person['race'], search_person['personSearch'], color_discrete_sequence=px.colors.qualitative.Pastel)

fig.update_layout(title={
                  'text': "Person Search based on Race",
                  'y':0.98,
                  'x':0.5,
                  'xanchor': 'center',
                  'yanchor': 'top'},
                   height=600,
                  template='plotly_white')

fig.update_traces(textposition='inside', textinfo='percent+label', pull=[0, 0.2])

fig.data[0].marker.line.width = 1
fig.data[0].marker.line.color = "black"

fig.show()


# In[21]:


vehicleSearch_df = stop_data[stop_data['vehicleSearch'] == 'YES']

search_vehicle = vehicleSearch_df.groupby('race')['vehicleSearch'].count().reset_index()

search_vehicle.sort_values(by='vehicleSearch', ascending=False).style.background_gradient(cmap='Oranges', subset=['vehicleSearch'])


# In[22]:


fig = px.pie(search_vehicle, search_vehicle['race'], search_vehicle['vehicleSearch'], color_discrete_sequence=px.colors.qualitative.Pastel)

fig.update_layout(title={
                  'text': "Vehicle Search based on Race",
                  'y':0.98,
                  'x':0.5,
                  'xanchor': 'center',
                  'yanchor': 'top'},
                   height=600,
                  template='plotly_white')

fig.update_traces(textposition='inside', textinfo='percent+label', pull=[0, 0.2])

fig.data[0].marker.line.width = 1
fig.data[0].marker.line.color = "black"

fig.show()


# In[23]:


fig = px.bar(stop_data['gender'].value_counts(), color_discrete_sequence = [px.colors.qualitative.Pastel])
fig.update_layout(title={
                  'text': "Police Stops Based on Gender",
                  'y':0.98,
                  'x':0.5,
                  'xanchor': 'center',
                  'yanchor': 'top'}, 
                  template='ggplot2', 
                  xaxis_title="Gender", 
                  yaxis_title="# Police Stops",
                  showlegend=False
                 )
fig.show()


# In[24]:


fig = px.bar(force_data['Race'].value_counts(), color_discrete_sequence = [px.colors.qualitative.Pastel])

fig.update_layout(title={
                  'text': "Police Violence by Race",
                  'y':0.98,
                  'x':0.5,
                  'xanchor': 'center',
                  'yanchor': 'top'}, 
                  template='ggplot2', 
                  xaxis_title="Race", 
                  yaxis_title="# Police Violence",
                  showlegend=False
                 )


fig.show()


# In[25]:


fig = px.pie(force_data, force_data['ForceType'], color_discrete_sequence=px.colors.qualitative.Pastel, hole=0.5)

fig.update_layout(title={
                  'text': "Type of Force by Police",
                  'y':0.98,
                  'x':0.5,
                  'xanchor': 'center',
                  'yanchor': 'top'},
                   height=600,
                  template='plotly_white')

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.data[0].marker.line.width = 1
fig.data[0].marker.line.color = "black"

fig.show()


# In[26]:


force_race = force_data.groupby(['Race'])[['ForceType']].count().reset_index()
force_race.sort_values(by='ForceType', ascending=False).style.background_gradient(cmap='summer', subset=['ForceType'])


# In[27]:


fig = px.pie(force_race, force_race['Race'], force_race['ForceType'], color_discrete_sequence=px.colors.qualitative.Pastel, hole=0.5)

fig.update_layout(title={
                  'text': "Force Type Based on Race",
                  'y':0.98,
                  'x':0.5,
                  'xanchor': 'center',
                  'yanchor': 'top'},
                   height=600,
                  template='plotly_white')

fig.update_traces(textposition='inside', textinfo='percent+label', pull=[0, 0.1])

fig.data[0].marker.line.width = 2
fig.data[0].marker.line.color = "black"

fig.show()


# In[28]:


fig = px.bar(force_data['ForceTypeAction'].value_counts(), color_discrete_sequence = px.colors.qualitative.Pastel)

fig.update_layout(title={
                  'text': "Action of Force Type",
                  'y':0.98,
                  'x':0.5,
                  'xanchor': 'center',
                  'yanchor': 'top'}, 
                  template='ggplot2', 
                  xaxis_title={'text': "Action"}, 
                  yaxis_title="# Police Violence",
                  height=1000,
                  showlegend=False
                 )

fig.show()


# In[29]:


white_action = force_data[force_data['Race'] == 'White']
fig = px.bar(white_action['ForceTypeAction'].value_counts(), color_discrete_sequence = px.colors.qualitative.Pastel)

fig.update_layout(title={
                  'text': "Action of Force Type on White People",
                  'y':0.98,
                  'x':0.5,
                  'xanchor': 'center',
                  'yanchor': 'top'}, 
                  template='ggplot2', 
                  xaxis_title={'text': "Action"}, 
                  yaxis_title="# Police Violence",
                  height=1000,
                  showlegend=False
                 )

fig.show()


# In[30]:


white_action['ForceTypeAction'].value_counts()


# In[31]:


black_action = force_data[force_data['Race'] == 'Black']
fig = px.bar(black_action['ForceTypeAction'].value_counts(), color_discrete_sequence = px.colors.qualitative.Pastel)

fig.update_layout(title={
                  'text': "Action of Force Type on Black People",
                  'y':0.98,
                  'x':0.5,
                  'xanchor': 'center',
                  'yanchor': 'top'}, 
                  template='ggplot2', 
                  xaxis_title={'text': "Action"}, 
                  yaxis_title="# Police Violence",
                  height=1000,
                  showlegend=False
                 )

fig.show()


# In[32]:


black_action['ForceTypeAction'].value_counts()


# In[33]:


fig = px.bar(force_data['Problem'].value_counts(), color_discrete_sequence = px.colors.qualitative.Pastel)

fig.update_layout(title={
                  'text': "Problem Reported by Police",
                  'y':0.98,
                  'x':0.5,
                  'xanchor': 'center',
                  'yanchor': 'top'}, 
                  template='ggplot2', 
                  xaxis_title="Problem", 
                  yaxis_title="Count",
                  height=700,
                  showlegend=False
                 )

fig.show()


# In[34]:


fig = px.pie(force_race, force_data['SubjectInjury'], color_discrete_sequence=px.colors.qualitative.Pastel, hole=0.5)

fig.update_layout(title={
                  'text': "# Times Subject Was Injured",
                  'y':0.98,
                  'x':0.5,
                  'xanchor': 'center',
                  'yanchor': 'top'},
                   height=600,
                  template='plotly_white')

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.data[0].marker.line.width = 2
fig.data[0].marker.line.color = "black"

fig.show()


# In[35]:


fig = px.pie(force_data, force_data['Sex'], color_discrete_sequence=px.colors.qualitative.Pastel, hole=0.5)

fig.update_layout(title={
                  'text': "Gender of Subject",
                  'y':0.98,
                  'x':0.5,
                  'xanchor': 'center',
                  'yanchor': 'top'},
                   height=600,
                  template='plotly_white')

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.data[0].marker.line.width = 2
fig.data[0].marker.line.color = "black"

fig.show()


# In[36]:


fig = px.pie(force_data, force_data['EventAge'], color_discrete_sequence=px.colors.qualitative.Pastel, hole=0.6)

fig.update_layout(title={
                  'text': "Age of Subject",
                  'y':0.98,
                  'x':0.5,
                  'xanchor': 'center',
                  'yanchor': 'top'},
                   height=600,
                  template='plotly_white', 
                  showlegend=False)

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.data[0].marker.line.width = 2
fig.data[0].marker.line.color = "black"

fig.show()


# In[37]:


from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


# In[40]:


# 转换 preRace、race 和 gender 列为哑变量
df = pd.get_dummies(df, columns=['preRace', 'race', 'gender'], drop_first=True)


# In[41]:


# 打印数据集的前几行
print(df.head())


# In[42]:


# 打印转换后的数据集信息
print(df.info())


# In[43]:


# 将布尔型数据转换为数值型
bool_cols = df.select_dtypes(include=['bool']).columns
df[bool_cols] = df[bool_cols].astype(int)


# In[44]:


# 打印转换后的数据集信息
print(df.info())


# In[45]:


# 打印转换后的数据集的前几行
print(df.head())


# In[46]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# In[47]:


# 假设你的数据集存储在df_clean
#df['stop'] = 'YES'

# 打印前几行数据以验证
#print(df.head())


# In[48]:


# 转换 personSearch 和 vehicleSearch 为数值类型
df['personSearch'] = df['personSearch'].map({'YES': 1, 'NO': 0})
df['vehicleSearch'] = df['vehicleSearch'].map({'YES': 1, 'NO': 0})
#df['stop'] = df['stop'].map({'YES': 1, 'NO': 0})

# 检查转换后的结果
print(df[['personSearch', 'vehicleSearch']].head())


# In[49]:


# 定义特征变量
features = [
    'preRace_Black', 'preRace_East African', 'preRace_Latino', 'preRace_Native American',
    'preRace_Other', 'preRace_Unknown', 'preRace_White', 'race_Black', 'race_East African',
    'race_Latino', 'race_Native American', 'race_Other', 'race_Unknown', 'race_White'
]


# In[50]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_curve

# 删除包含缺失值的行
df_clean = df.dropna(subset=['personSearch', 'vehicleSearch'])

# 遍历每个目标变量
for target_column in ['personSearch', 'vehicleSearch']:
    # 定义特征矩阵 X 和目标向量 y
    X = df_clean[features]
    y = df_clean[target_column]

    # 标准化特征矩阵 X
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
   
    # 应用PCA
    pca = PCA(n_components=0.95)  # 保留至少95%的方差解释比例
    X_pca = pca.fit_transform(X_scaled)

    # 拆分数据集为训练集和测试集
    X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

    # 创建逻辑回归模型
    logreg_pca = LogisticRegression(max_iter=1000, class_weight='balanced')

    # 训练模型
    logreg_pca.fit(X_train_pca, y_train)

    # 预测概率
    y_pred_proba = logreg_pca.predict_proba(X_test_pca)[:, 1]

    # 找到最佳阈值
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    f1_scores = 2 * recall * precision / (recall + precision)
    best_threshold = thresholds[np.argmax(f1_scores)]

    # 打印不同阈值的性能
    for threshold in np.arange(0.1, 1.0, 0.1):
        y_pred_threshold = (y_pred_proba >= threshold).astype(int)
        accuracy = accuracy_score(y_test, y_pred_threshold)
        conf_matrix = confusion_matrix(y_test, y_pred_threshold)
        class_report = classification_report(y_test, y_pred_threshold)
        
        print(f"Threshold: {threshold}")
        print(f"Accuracy: {accuracy}")
        print(f"Confusion Matrix:\n{conf_matrix}")
        print(f"Classification Report:\n{class_report}")
        print("="*50)

    # 用最佳阈值进行预测
    y_pred_best_threshold = (y_pred_proba >= best_threshold).astype(int)

    # 评估新阈值的模型
    accuracy = accuracy_score(y_test, y_pred_best_threshold)
    conf_matrix = confusion_matrix(y_test, y_pred_best_threshold)
    class_report = classification_report(y_test, y_pred_best_threshold)

    print(f"Results for target column '{target_column}':")
    print(f"Adjusted Threshold: {best_threshold}")
    print(f"Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Classification Report:\n{class_report}")

    # 输出模型系数
    coefficients = pd.DataFrame(logreg_pca.coef_.flatten(), columns=['Coefficient'])
    print(coefficients)

    # 输出PCA的解释方差比
    print(f"Explained variance ratio by PCA: {pca.explained_variance_ratio_}")


# In[51]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_recall_curve
import joblib

# 删除包含缺失值的行
df_clean = df.dropna(subset=['personSearch', 'vehicleSearch'])

# 遍历每个目标变量
for target_column in ['personSearch', 'vehicleSearch']:
    # 定义特征矩阵 X 和目标向量 y
    X = df_clean[features]
    y = df_clean[target_column]

    # 标准化特征矩阵 X
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
   
    # 应用PCA
    pca = PCA(n_components=0.95)  # 保留至少95%的方差解释比例
    X_pca = pca.fit_transform(X_scaled)

    # 拆分数据集为训练集和测试集
    X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

    # 创建逻辑回归模型
    logreg_pca = LogisticRegression(max_iter=1000, class_weight='balanced')

    # 训练模型
    logreg_pca.fit(X_train_pca, y_train)

    # 预测概率
    y_pred_proba = logreg_pca.predict_proba(X_test_pca)[:, 1]

    # 找到最佳阈值
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    f1_scores = 2 * recall * precision / (recall + precision)
    best_threshold = thresholds[np.argmax(f1_scores)]

    # 用最佳阈值进行预测
    y_pred_best_threshold = (y_pred_proba >= best_threshold).astype(int)

    # 评估新阈值的模型
    accuracy = accuracy_score(y_test, y_pred_best_threshold)
    conf_matrix = confusion_matrix(y_test, y_pred_best_threshold)
    class_report = classification_report(y_test, y_pred_best_threshold)

    print(f"Results for target column '{target_column}':")
    print(f"Adjusted Threshold: {best_threshold}")
    print(f"Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Classification Report:\n{class_report}")

    # 保存模型、scaler、pca和最佳阈值
    joblib.dump(logreg_pca, f'{target_column}_logreg_pca.pkl')
    joblib.dump(scaler, f'{target_column}_scaler.pkl')
    joblib.dump(pca, f'{target_column}_pca.pkl')
    joblib.dump(best_threshold, f'{target_column}_best_threshold.pkl')


# In[52]:


import numpy as np
import joblib
import pandas as pd

def predict_risk(preRace_features, race_features):
    # 合并 preRace 和 race 的特征
    input_data = pd.DataFrame([preRace_features + race_features], columns=features)

    results = {}

    for target_column in ['personSearch', 'vehicleSearch']:
        # 加载模型、scaler、pca和最佳阈值
        logreg_pca = joblib.load(f'{target_column}_logreg_pca.pkl')
        scaler = joblib.load(f'{target_column}_scaler.pkl')
        pca = joblib.load(f'{target_column}_pca.pkl')
        best_threshold = joblib.load(f'{target_column}_best_threshold.pkl')

        # 标准化输入数据
        input_scaled = scaler.transform(input_data)
       
        # 应用PCA
        input_pca = pca.transform(input_scaled)

        # 预测概率
        risk_score = logreg_pca.predict_proba(input_pca)[:, 1][0]

        # 判断是否要进行personSearch或vehicleSearch
        decision = 'YES' if risk_score >= best_threshold else 'NO'

        results[target_column] = {
            'risk_score': risk_score,
            'decision': decision
        }

    return results

# 示例输入
#'preRace_Black', 'preRace_East African', 'preRace_Latino', 'preRace_Native American', 'preRace_Other', 'preRace_Unknown', 'preRace_White';
#'race_Black', 'race_East African','race_Latino', 'race_Native American', 'race_Other', 'race_Unknown', 'race_White';
preRace_features = [1, 0, 0, 0, 0, 0, 0]  # 替换为实际的输入值，例如 preRace_Black = 1
race_features = [0, 0, 1, 0, 0, 0, 0]     # 替换为实际的输入值，例如 race_Latino = 1
result = predict_risk(preRace_features, race_features)
print(result)


# In[53]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV


# In[58]:


# 遍历每个目标变量
for target_column in ['personSearch', 'vehicleSearch']:
    # 定义特征矩阵 X 和目标向量 y
    X = df_clean[features]
    y = df_clean[target_column]

    # 标准化特征矩阵 X
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 拆分数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 定义随机森林模型和参数网格
    rf = RandomForestClassifier(class_weight='balanced', random_state=42)
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # 使用网格搜索进行超参数调优
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='f1', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # 获取最佳模型
    best_rf = grid_search.best_estimator_

    # 预测概率
    y_pred_proba = best_rf.predict_proba(X_test)[:, 1]

    # 找到最佳阈值
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    f1_scores = 2 * recall * precision / (recall + precision)
    best_threshold = thresholds[np.argmax(f1_scores)]

    # 用最佳阈值进行预测
    y_pred_best_threshold = (y_pred_proba >= best_threshold).astype(int)

    # 评估新阈值的模型
    accuracy = accuracy_score(y_test, y_pred_best_threshold)
    conf_matrix = confusion_matrix(y_test, y_pred_best_threshold)
    class_report = classification_report(y_test, y_pred_best_threshold)

    print(f"Results for target column '{target_column}':")
    print(f"Best Hyperparameters: {grid_search.best_params_}")
    print(f"Adjusted Threshold: {best_threshold}")
    print(f"Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Classification Report:\n{class_report}")

    # 输出特征重要性
    feature_importances = pd.DataFrame(best_rf.feature_importances_, index=features, columns=['Importance']).sort_values(by='Importance', ascending=False)
    print("Feature Importances:")
    print(feature_importances)
     # 保存模型和特征
    model_filename = f'best_rf_model_{target_column}.joblib'
    scaler_filename = f'scaler_{target_column}.joblib'
    threshold_filename = f'best_threshold_{target_column}.npy'
    features_filename = f'features_{target_column}.npy'

    joblib.dump(best_rf, model_filename)
    joblib.dump(scaler, scaler_filename)
    np.save(threshold_filename, best_threshold)
    np.save(features_filename, np.array(features))

    print(f"Model, scaler, threshold, and features for '{target_column}' saved successfully.")


# In[59]:


# 加载模型和其他文件
best_rf_personSearch = joblib.load('best_rf_model_personSearch.joblib')
scaler_personSearch = joblib.load('scaler_personSearch.joblib')
best_threshold_personSearch = np.load('best_threshold_personSearch.npy')
features_personSearch = np.load('features_personSearch.npy').tolist()

best_rf_vehicleSearch = joblib.load('best_rf_model_vehicleSearch.joblib')
scaler_vehicleSearch = joblib.load('scaler_vehicleSearch.joblib')
best_threshold_vehicleSearch = np.load('best_threshold_vehicleSearch.npy')
features_vehicleSearch = np.load('features_vehicleSearch.npy').tolist()

# 对新数据进行预测
new_data = pd.DataFrame({
    'race_Black': [0, 1, 0],  # 请根据实际数据替换
    'race_Unknown': [0, 0, 1],
    'race_White': [1, 0, 0],
    'preRace_White': [1, 0, 0],
    'race_Native American': [0, 0, 0],
    'preRace_Unknown': [0, 0, 1],
    'preRace_Black': [0, 1, 0],
    'race_East African': [0, 0, 0],
    'race_Latino': [0, 0, 1],
    'preRace_Native American': [0, 0, 0],
    'race_Other': [0, 0, 0],
    'preRace_Latino': [0, 0, 1],
    'preRace_Other': [0, 0, 0],
    'preRace_East African': [0, 0, 0]
})

# 确保新数据包含相应的特征
assert all([col in new_data.columns for col in features_personSearch]), "New data does not contain all required features for personSearch."
assert all([col in new_data.columns for col in features_vehicleSearch]), "New data does not contain all required features for vehicleSearch."

# 标准化新数据
new_data_scaled_personSearch = scaler_personSearch.transform(new_data[features_personSearch])
new_data_scaled_vehicleSearch = scaler_vehicleSearch.transform(new_data[features_vehicleSearch])

# 预测概率
risk_scores_personSearch = best_rf_personSearch.predict_proba(new_data_scaled_personSearch)[:, 1]
risk_scores_vehicleSearch = best_rf_vehicleSearch.predict_proba(new_data_scaled_vehicleSearch)[:, 1]

# 根据最佳阈值预测类别
predictions_personSearch = (risk_scores_personSearch >= best_threshold_personSearch).astype(int)
predictions_vehicleSearch = (risk_scores_vehicleSearch >= best_threshold_vehicleSearch).astype(int)

print("Risk scores for personSearch:")
print(risk_scores_personSearch)
print("Predictions for personSearch:")
print(predictions_personSearch)

print("Risk scores for vehicleSearch:")
print(risk_scores_vehicleSearch)
print("Predictions for vehicleSearch:")
print(predictions_vehicleSearch)


# In[60]:


import numpy as np
import pandas as pd
import joblib

# 加载模型和其他文件
best_rf_personSearch = joblib.load('best_rf_model_personSearch.joblib')
scaler_personSearch = joblib.load('scaler_personSearch.joblib')
best_threshold_personSearch = np.load('best_threshold_personSearch.npy')
features_personSearch = np.load('features_personSearch.npy').tolist()

best_rf_vehicleSearch = joblib.load('best_rf_model_vehicleSearch.joblib')
scaler_vehicleSearch = joblib.load('scaler_vehicleSearch.joblib')
best_threshold_vehicleSearch = np.load('best_threshold_vehicleSearch.npy')
features_vehicleSearch = np.load('features_vehicleSearch.npy').tolist()

def predict_search(preRace, race):
    # 构建输入数据
    input_data = {
        'race_Black': 0,  
        'race_Unknown': 0,
        'race_White': 0,
        'preRace_White': 0,
        'race_Native American': 0,
        'preRace_Unknown': 0,
        'preRace_Black': 0,
        'race_East African': 0,
        'race_Latino': 0,
        'preRace_Native American': 0,
        'race_Other': 0,
        'preRace_Latino': 0,
        'preRace_Other': 0,
        'preRace_East African': 0
    }
    
    input_data[f'preRace_{preRace}'] = 1
    input_data[f'race_{race}'] = 1
    
    new_data = pd.DataFrame([input_data])
    
    # 标准化新数据
    new_data_scaled_personSearch = scaler_personSearch.transform(new_data[features_personSearch])
    new_data_scaled_vehicleSearch = scaler_vehicleSearch.transform(new_data[features_vehicleSearch])
    
    # 预测概率
    risk_scores_personSearch = best_rf_personSearch.predict_proba(new_data_scaled_personSearch)[:, 1]
    risk_scores_vehicleSearch = best_rf_vehicleSearch.predict_proba(new_data_scaled_vehicleSearch)[:, 1]
    
    # 根据最佳阈值预测类别
    prediction_personSearch = (risk_scores_personSearch >= best_threshold_personSearch).astype(int)
    prediction_vehicleSearch = (risk_scores_vehicleSearch >= best_threshold_vehicleSearch).astype(int)
    
    results = {
        "Risk Score for Person Search": risk_scores_personSearch[0],
        "Prediction for Person Search": "Yes" if prediction_personSearch[0] == 1 else "No",
        "Risk Score for Vehicle Search": risk_scores_vehicleSearch[0],
        "Prediction for Vehicle Search": "Yes" if prediction_vehicleSearch[0] == 1 else "No"
    }
    
    return results

# 示例输入
preRace = 'Black'
race = 'Native American'
results = predict_search(preRace, race)
print(results)


# In[8]:


# 读取 CSV 文件到 DataFrame 中
df = pd.read_csv('police_stop_data.csv')
#转换 preRace、race 和 gender 列为哑变量
df = pd.get_dummies(df, columns=['preRace', 'race', 'gender'], drop_first=True)
# 打印数据集的前几行
print(df.head())
# 打印转换后的数据集信息
print(df.info())
# 将布尔型数据转换为数值型
bool_cols = df.select_dtypes(include=['bool']).columns
df[bool_cols] = df[bool_cols].astype(int)
# 打印转换后的数据集信息
print(df.info())
# 打印转换后的数据集的前几行
print(df.head())
# 转换 personSearch 和 vehicleSearch 为数值类型
df['personSearch'] = df['personSearch'].map({'YES': 1, 'NO': 0})
df['vehicleSearch'] = df['vehicleSearch'].map({'YES': 1, 'NO': 0})
#df['stop'] = df['stop'].map({'YES': 1, 'NO': 0})

# 检查转换后的结果
print(df[['personSearch', 'vehicleSearch']].head())


# In[11]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve, accuracy_score, confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
import joblib


# In[12]:


# 删除包含缺失值的行
df_clean = df.dropna(subset=['personSearch', 'vehicleSearch'])
# 遍历每个目标变量
for target_column in ['personSearch', 'vehicleSearch']:
    # 定义特征矩阵 X 和目标向量 y
    X = df_clean[features]
    y = df_clean[target_column]

    # 标准化特征矩阵 X
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 拆分数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 定义神经网络模型
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # 编译模型
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 定义模型保存的回调
    checkpoint = ModelCheckpoint(f'best_nn_model_{target_column}.keras', monitor='val_loss', save_best_only=True, mode='min')

    # 训练模型
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[checkpoint], verbose=1)

    # 预测概率
    y_pred_proba = model.predict(X_test).flatten()

    # 找到最佳阈值
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    f1_scores = 2 * recall * precision / (recall + precision)
    best_threshold = thresholds[np.argmax(f1_scores)]

    # 用最佳阈值进行预测
    y_pred_best_threshold = (y_pred_proba >= best_threshold).astype(int)

    # 评估新阈值的模型
    accuracy = accuracy_score(y_test, y_pred_best_threshold)
    conf_matrix = confusion_matrix(y_test, y_pred_best_threshold)
    class_report = classification_report(y_test, y_pred_best_threshold)

    print(f"Results for target column '{target_column}':")
    print(f"Adjusted Threshold: {best_threshold}")
    print(f"Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Classification Report:\n{class_report}")

    # 保存缩放器和最佳阈值
    joblib.dump(scaler, f'scaler_{target_column}.joblib')
    np.save(f'best_threshold_{target_column}.npy', best_threshold)
    np.save(f'features_{target_column}.npy', features)


# In[13]:


import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# 加载模型和其他文件
model_personSearch = load_model('best_nn_model_personSearch.keras')
scaler_personSearch = joblib.load('scaler_personSearch.joblib')
best_threshold_personSearch = np.load('best_threshold_personSearch.npy')
features_personSearch = np.load('features_personSearch.npy').tolist()

model_vehicleSearch = load_model('best_nn_model_vehicleSearch.keras')
scaler_vehicleSearch = joblib.load('scaler_vehicleSearch.joblib')
best_threshold_vehicleSearch = np.load('best_threshold_vehicleSearch.npy')
features_vehicleSearch = np.load('features_vehicleSearch.npy').tolist()

def predict_search(preRace, race):
    # 构建输入数据
    input_data = {
        'race_Black': 0,  
        'race_Unknown': 0,
        'race_White': 0,
        'preRace_White': 0,
        'race_Native American': 0,
        'preRace_Unknown': 0,
        'preRace_Black': 0,
        'race_East African': 0,
        'race_Latino': 0,
        'preRace_Native American': 0,
        'race_Other': 0,
        'preRace_Latino': 0,
        'preRace_Other': 0,
        'preRace_East African': 0
    }
    
    input_data[f'preRace_{preRace}'] = 1
    input_data[f'race_{race}'] = 1
    
    new_data = pd.DataFrame([input_data])
    
    # 标准化新数据
    new_data_scaled_personSearch = scaler_personSearch.transform(new_data[features_personSearch])
    new_data_scaled_vehicleSearch = scaler_vehicleSearch.transform(new_data[features_vehicleSearch])
    
    # 预测概率
    risk_scores_personSearch = model_personSearch.predict(new_data_scaled_personSearch).flatten()
    risk_scores_vehicleSearch = model_vehicleSearch.predict(new_data_scaled_vehicleSearch).flatten()
    
    # 根据最佳阈值预测类别
    prediction_personSearch = (risk_scores_personSearch >= best_threshold_personSearch).astype(int)
    prediction_vehicleSearch = (risk_scores_vehicleSearch >= best_threshold_vehicleSearch).astype(int)
    
    results = {
        "Risk Score for Person Search": risk_scores_personSearch[0],
        "Prediction for Person Search": "Yes" if prediction_personSearch[0] == 1 else "No",
        "Risk Score for Vehicle Search": risk_scores_vehicleSearch[0],
        "Prediction for Vehicle Search": "Yes" if prediction_vehicleSearch[0] == 1 else "No"
    }
    
    return results

# 示例输入
preRace = 'Black'
race = 'Native American'
results = predict_search(preRace, race)
print(results)


# In[15]:


for target_column in ['personSearch', 'vehicleSearch']:
    # 定义特征矩阵 X 和目标向量 y
    X = df_clean[features]
    y = df_clean[target_column]

    # 标准化特征矩阵 X
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 拆分数据集为训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 构建神经网络模型
    model = Sequential()
    model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # 编译模型
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 定义模型保存的回调
    checkpoint = ModelCheckpoint(f'best_nn_model_{target_column}.keras', monitor='val_loss', save_best_only=True, mode='min')

    # 训练模型
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[checkpoint], verbose=1)

    # 加载最佳模型
    model.load_weights(f'best_nn_model_{target_column}.keras')

    # 预测概率
    y_pred_proba = model.predict(X_test).flatten()

    # 找到最佳阈值
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)
    f1_scores = 2 * recall * precision / (recall + precision)
    best_threshold = thresholds[np.argmax(f1_scores)]

    # 用最佳阈值进行预测
    y_pred_best_threshold = (y_pred_proba >= best_threshold).astype(int)

    # 评估新阈值的模型
    accuracy = accuracy_score(y_test, y_pred_best_threshold)
    conf_matrix = confusion_matrix(y_test, y_pred_best_threshold)
    class_report = classification_report(y_test, y_pred_best_threshold)

    print(f"Results for target column '{target_column}':")
    print(f"Adjusted Threshold: {best_threshold}")
    print(f"Accuracy: {accuracy}")
    print(f"Confusion Matrix:\n{conf_matrix}")
    print(f"Classification Report:\n{class_report}")

    # 保存模型和标量
    model.save(f'best_nn_model_{target_column}.keras')
    joblib.dump(scaler, f'scaler_{target_column}.joblib')
    np.save(f'best_threshold_{target_column}.npy', best_threshold)
    np.save(f'features_{target_column}.npy', np.array(features))


# In[18]:


import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# 加载模型和其他文件
model_personSearch = load_model('best_nn_model_personSearch.keras')
scaler_personSearch = joblib.load('scaler_personSearch.joblib')
best_threshold_personSearch = np.load('best_threshold_personSearch.npy')
features_personSearch = np.load('features_personSearch.npy').tolist()

model_vehicleSearch = load_model('best_nn_model_vehicleSearch.keras')
scaler_vehicleSearch = joblib.load('scaler_vehicleSearch.joblib')
best_threshold_vehicleSearch = np.load('best_threshold_vehicleSearch.npy')
features_vehicleSearch = np.load('features_vehicleSearch.npy').tolist()

def predict_search(preRace, race):
    # 构建输入数据
    input_data = {
        'race_Black': 0,  
        'race_Unknown': 0,
        'race_White': 0,
        'preRace_White': 0,
        'race_Native American': 0,
        'preRace_Unknown': 0,
        'preRace_Black': 0,
        'race_East African': 0,
        'race_Latino': 0,
        'preRace_Native American': 0,
        'race_Other': 0,
        'preRace_Latino': 0,
        'preRace_Other': 0,
        'preRace_East African': 0
    }
    
    input_data[f'preRace_{preRace}'] = 1
    input_data[f'race_{race}'] = 1
    
    new_data = pd.DataFrame([input_data])
    
    # 标准化新数据
    new_data_scaled_personSearch = scaler_personSearch.transform(new_data[features_personSearch])
    new_data_scaled_vehicleSearch = scaler_vehicleSearch.transform(new_data[features_vehicleSearch])
    
    # 预测概率
    risk_scores_personSearch = model_personSearch.predict(new_data_scaled_personSearch).flatten()
    risk_scores_vehicleSearch = model_vehicleSearch.predict(new_data_scaled_vehicleSearch).flatten()
    
    # 根据最佳阈值预测类别
    prediction_personSearch = (risk_scores_personSearch >= best_threshold_personSearch).astype(int)
    prediction_vehicleSearch = (risk_scores_vehicleSearch >= best_threshold_vehicleSearch).astype(int)
    
    results = {
        "Risk Score for Person Search": risk_scores_personSearch[0],
        "Prediction for Person Search": "Yes" if prediction_personSearch[0] == 1 else "No",
        "Risk Score for Vehicle Search": risk_scores_vehicleSearch[0],
        "Prediction for Vehicle Search": "Yes" if prediction_vehicleSearch[0] == 1 else "No"
    }
    
    return results

# 示例输入
preRace = 'White'
race = 'Black'
results = predict_search(preRace, race)
print(results)

