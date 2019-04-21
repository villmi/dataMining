from sklearn.datasets import load_iris

# sklearn围绕着机器学习提供了很多可直接调用的机器学习算法以及很多经典的数据集.存放在datasets模块中
iris = load_iris()
# print(iris.data.shape) # 查验数据规模
# print(iris.DESCR) #查看数据说明

# 从sklearn.cross_validation里选择导入train_test_split用于数据分割
# from sklearn.cross_validation import train_test_split
import sklearn.cross_decomposition

# 使用train_test_split,利用随机种子random_state采样25%的数据作为测试集
# x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.25, random_state=33)
x_train, x_test, y_train, y_test = sklearn.cross_decomposition.train_test_split(iris.data, iris.target, test_size=0.25, random_state=33)

# 使用K近邻分类器对Iris数据进行类别预测
# 从sklearn.preprocessing里选择导入数据标准化模块
from sklearn.preprocessing import StandardScaler
# 从sklearn.neighbors里选择导入KNeighborsClassifier,即K近邻分类器
from sklearn.neighbors import KNeighborsClassifier

# 对训练和测试的特征数据进行标准化
ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)
knc = KNeighborsClassifier()
knc.fit(x_train, y_train)
y_predict = knc.predict(x_test)
print("The accuracy of K-Nearest Neighbor Classifier is", knc.score(x_test, y_test))

from sklearn.metrics import classification_report

print(classification_report(y_test, y_predict, target_names=iris.target_names))
