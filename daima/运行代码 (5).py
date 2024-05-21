import pandas as pd
import numpy as np
#通过警告过滤器控制是否发出警告  在这是忽略警告
import warnings
warnings.filterwarnings("ignore")


# 读取数据
inputfile = '../dataa/shuju.csv' # 输入的数据文件
data = pd.read_csv(inputfile) # 读取数据

############################################## 哪些球队最热衷于参加友谊赛(聚类分析) ##########################################

# 统计home_team每个队伍参加的次数
mid_1 = data["home_team"].value_counts() #对重复元素进行统计
print('home_team的队伍及参赛次数\n\n',mid_1,'\n\n') #输出队伍和进行比赛的次数


# 统计away_team每个队伍参加的次数
mid_2 = data["away_team"].value_counts()
print('away_team的队伍及参赛次数\n\n',mid_2,'\n\n')


# 两个表格合并
indexs = [] #定义一个数组
for i in mid_2.index.tolist(): #遍历列表（.index.tolist()将索引转换为列表）中所有元素
    if i in mid_1.index.tolist():
        indexs.append(i) #将遍历到的数据存入数组中


data_team = pd.DataFrame([mid_1[indexs].tolist(),mid_2[indexs].tolist()],index = ["home_team","away_team"]).T
data_team.index=mid_1[indexs].index #聚类索引为mid_1这一列
print('将两个表格合并\n\n',data_team,'\n\n') #将合并的表输出


# 标准化数据
from sklearn.preprocessing import StandardScaler #sklearn数据特征预处理（标准化）
scaler = StandardScaler()
scaler.fit(data_team) #用于计算训练数据的均值和方差
X_scaled = scaler.transform(data_team) #用得到的均值和方差来转换数据，使其标准化


# 标准化表格存储
Scaled = pd.DataFrame(X_scaled,columns=data_team.columns,index=data_team.index)

# 空值用0填充
Scaled = Scaled.fillna(0)

# 查看标准化结果
print('标准化后的数据\n\n',Scaled,'\n\n')


# kmeans聚类训练
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# 下面是用K-Means聚类方法做的聚类
mod = KMeans(n_clusters=4,random_state=255) #调用聚类模型 聚类为4类
print('构建的KM模型为\n\n',mod,'\n\n')
y_pre = mod.fit_predict(Scaled)
print('构建的KM模型的结果为\n\n',y_pre,'\n\n')
# 将结果存于表格
data_team["kmeans"] = y_pre

# 可视化查看聚类结果(1类球队最热衷于参加友谊赛)
plt.figure(figsize=(10,6)) #设定空白画布，并制定大小
# sns.scatterplot()画散点图
# x,y为数据中变量的名称 作用是生成具有不同颜色的元素的变量进行分组
# hue 根据设置的类别，产生颜色不同的点的散点图 根据kmeans进行的分类


sns.scatterplot(x="home_team",y='away_team',data=data_team,hue='kmeans')
# plt.scatter()画散点图
# x,y是数组，即即将绘制散点图的数据点
# s是数组的大小，即散点的点的大小
# c表示颜色   marker表示标记的样式

plt.scatter(data_team.groupby("kmeans").mean().iloc[:,0],data_team.groupby("kmeans").mean().iloc[:,1],c="k",marker="*",s=150)
plt.show()


# 这些队伍最爱参加友谊赛
print("这些队伍最爱参加友谊赛：\n")
print(' ',data_team[data_team["kmeans"]==1].index,'  ')


#################################################### 可否进行比赛成绩预测(分类模型) ###########################################

# 将时间设置为索引
data = data.set_index("date")


# 制作特征，主队是否获胜，获胜为1，否则为0
result = []
for i in range(len(data)):
    if data.iloc[i,2]>data.iloc[i,3]: #iloc函数是通过行号获取行数据
        result.append(1)
    else:
        result.append(0)

# 加入数据
data["result"] = result

# 将neutral类型转换为str
data["neutral"] = data["neutral"].astype("str")

# 查看数据类型
print(' \n\n',data.info(),'\n\n')

# 将类型为object类型的标签编码为数值类型
from sklearn.preprocessing import LabelEncoder
data.iloc[:,[0,1,4,5,6,7]] = data.iloc[:,[0,1,4,5,6,7]].apply(LabelEncoder().fit_transform)
# 将数据类型都转换为float类型用于建模
data = data.apply(lambda x:x.astype(float))

# 查看转换之后的数据类型看是否有异常
print('\n\n',data.info(),'\n\n')

# 查看数据正负样本是否均衡
import matplotlib.pyplot as plt
# 调试图例
plt.figure(figsize=(6,6))
# 设置标签
labels = data["result"].value_counts().index
# 设置扇形间隔
explode=(0.1,0.1)
# 绘制饼图
plt.pie(data["result"].value_counts().tolist(),explode=explode,labels=labels,
autopct='%3.1f%%',#文本标签对应的数值百分比样式
startangle=45,#x轴起始位置,第一个饼片逆时针旋转的角度
shadow=True)
# 设置标题
plt.title("Proportion of various types")
plt.show()

# 特征，删除标签列和分数
X = data.drop(["away_score","home_score","result"],axis=1)
# 标签
y = data["result"]

# 相关度（特征直接没有较大的相关性不需要pca降维）
corr = X.corr()
plt.figure(figsize=(10,10))

# 相关度热力图
sns.heatmap(corr, cmap='GnBu_r', square=True, annot=True)
plt.show()

# train_test_split返回划分的训练集train/测试集test
from sklearn.model_selection import train_test_split
# 切分数据，0.2做测试集，0.8做训练集
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.2, random_state=255)

from sklearn.model_selection import GridSearchCV #网格搜索
from sklearn.tree import DecisionTreeClassifier
# DecisionTreeClassifier模型
tree_modle = DecisionTreeClassifier(random_state=255)
# 参数列表
tree_param = {"splitter":["best","random"],
              "max_depth":[5,10,15,20,30,50,80],
              "min_samples_leaf":[30,50,100]}
# 五折网格搜索法
clf = GridSearchCV(tree_modle,tree_param,cv = 5) #cv交叉验证参数
clf.fit(X_train, y_train) #运行网格搜索
# 打印参数
print("五折网格搜索法\n",clf.best_estimator_,"\n\n") #通过搜索选择的估计器，即在左侧数据上给出最高分数（或指定的最小损失）的估计器,估计器括号里包括选中的参数。如果refit = False，则不可用。

# 决策树模型
from sklearn.tree import DecisionTreeClassifier
dt_model = DecisionTreeClassifier(random_state=255) #调用决策树模型
# 训练模型
dt_model.fit(X_train, y_train) #用决策树拟合构造的数据集
# 预测
y_pre = dt_model.predict(X_test) #在训练集和测试集上分别用训练好的模型进行预测

#from sklearn.tree import  export_graphviz
#import graphviz
#dot_data=export_graphviz(dt_model,out_file=None)
#graph=graphviz.Source(dot_data)
#graph.render("决策树可视化")

features=X.columns #获取特征名称
importance=dt_model.feature_importances_ #获取特征重要性
#通过二维表格形式显示
importance_df=pd.DataFrame([features,importance],index=['特征名称','特征重要性']).T
print('特征重要性评估\n\n',importance_df,'\n\n')

from sklearn import metrics
# 混淆矩阵
# 查看混淆矩阵（预测值和真实值的各类情况统计矩阵）
cm = metrics.confusion_matrix(y_test,y_pre,labels=[0,1])
print('hhhhh\n',cm,'\n\n')

# 利用热力图对于结果进行可视化
sns.heatmap(cm,annot=True,cmap="GnBu")
plt.show()

# 各类评价指标
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pre),'\n\n')
#模型预测效果评估
from sklearn.metrics import roc_curve
fpr,tpr,thres=roc_curve(y_test,y_pre)
a=pd.DataFrame()
a['阙值']=list(thres)
a['假警报率']=list(fpr)
a['命中率']=list(tpr)
print('决策树模型的评估参数\n\n',a,'\n\n')
#绘制ROC曲线
import matplotlib.pyplot as plt
plt.plot(fpr,tpr)
plt.show()

# XGBClassifier模型
from xgboost import XGBClassifier
xgb_model = XGBClassifier(random_state=255)
print('XGBOOST构建的预测模型为\n\n',xgb_model,'\n\n')
# 训练模型
xgb_model.fit(X_train, y_train)
# 预测
y_pre = xgb_model.predict(X_test)

from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt

# 混淆矩阵
cm = metrics.confusion_matrix(y_test,y_pre,labels=[0,1])
sns.heatmap(cm,annot=True,cmap="GnBu")
plt.show()

# 各类评价指标
from sklearn.metrics import classification_report
print('mmmmm\n',classification_report(y_test,y_pre),'\n\n')

#模型预测效果评估
from sklearn.metrics import roc_curve
fpr,tpr,thres=roc_curve(y_test,y_pre)
a=pd.DataFrame()
a['阙值']=list(thres)
a['假警报率']=list(fpr)
a['命中率']=list(tpr)
print('xgboost模型的评估参数\n\n',a,'\n\n')
#绘制ROC曲线
import matplotlib.pyplot as plt
plt.plot(fpr,tpr)
plt.show()
# 决策树的准确率为0.58，xgboost的准确率为0.64，都高于0.5，所以是可以预测比赛输赢的，且xgboost模型比决策树模型要好