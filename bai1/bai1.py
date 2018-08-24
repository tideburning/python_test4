# Load libraries(2.1)
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset(2.2)
# data được load UCI machine learning
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = pandas.read_csv(url, names=names)

'''
#(3)
#Tóm tắt dữ liệu
# shape (3.1)
print('data set include 150 rows and 5 columns')
print(dataset.shape)

# head (3.2)
print('first 20 rows of the data set ')
print(dataset.head(20))

# descriptions(3.3)
print('Describe data by columns : number(so luong), mean( gia tri trung binh), min(gia tri nho nhat), max(gia tri lon nhat), ...')
print(dataset.describe())

# class distribution(3.4)
print('Describe a rows data, in this stuation is the column class')
print(dataset.groupby('class').size())
'''

#(4)
#Hiển thị dữ liệu
'''
#Các hộp đơn lẻ()
# box and whisker plots
#sharex duoc dung khi co nhieu truc
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
#plt.show()

#tạo biểu đồ của mỗi biến đầu vào để có ý tưởng về phân chia
# histograms
dataset.hist()
#plt.show()

# scatter plot matrix
# mạch ma trận phân tán
scatter_matrix(dataset)
plt.show()
'''

# Chia các bộ dữ liệu ra 2 phần (80% để train, 20% để test)
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 11
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Test options and evaluation metric
# chấm theo độ chính xác
scoring = 'accuracy' 

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
#print(models)


# evaluate each model in turn
results = []
names = []
for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

'''
# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
'''

# Make predictions on validation dataset
#knn = KNeighborsClassifier()
knn = SVC()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))