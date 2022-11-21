import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Đọc dữ liệu

data=pd.read_csv("healthcare-dataset-stroke-data.csv" )
x = data.iloc[:,1:11]
y = data.stroke
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3.0, random_state=100)

# Mô hình KNN

Mohinh_KNN = KNeighborsClassifier(n_neighbors=13)
Mohinh_KNN.fit(x_train, y_train)
y_pred_KNN = Mohinh_KNN.predict(x_test)
print("Accuracy is KNN: ", accuracy_score(y_test, y_pred_KNN)*100) # 95.305%

cnf_matrix_gnb = confusion_matrix(y_test,y_pred_KNN,labels=[1,0])
print(cnf_matrix_gnb)

# Bayes thơ ngây

model = GaussianNB()
model.fit(x_train, y_train)
y_pred_Bayes = model.predict(x_test)
print("Accuracy is Bayes: ", accuracy_score(y_test, y_pred_Bayes)*100) # 86.326%

# Cây quyết định

cayquyetdinh = DecisionTreeClassifier(criterion = 'gini', random_state = 100, max_depth = 7, min_samples_leaf = 5)
cayquyetdinh.fit(x_train,y_train)
y_pred_Tree = cayquyetdinh.predict(x_test)
print('Accuracy is DecisionTree: ', accuracy_score(y_test, y_pred_Tree)*100) # 94.953%



x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3.0, random_state=100)
acc=0
n=0
for k in [3,5,7,9,11,13,15,17,19]:
    Mohinh_KNN = KNeighborsClassifier(n_neighbors=k)
    Mohinh_KNN.fit(x_train, y_train)
    y_pred_KNN = Mohinh_KNN.predict(x_test)
    print("Giá trị K: ", k)
    print("Accuracy is KNN: ", accuracy_score(y_test, y_pred_KNN)*100)
    if(acc<=accuracy_score(y_test, y_pred_KNN)*100):
        acc=accuracy_score(y_test, y_pred_KNN)*100
        n=k

print("Accuracy max: ",acc)
print("Giá trị k có độ chính xác cao nhất: ",n) 



# Phần vẽ sơ đồ

arrKNN = []
arrBayes = []
arrTree = []
for i in range(0,10):
    X_train,X_test,y_train, y_test=train_test_split(x, y, test_size=1/3.0,random_state=100+i)
    Mohinh_KNN = KNeighborsClassifier(n_neighbors=101)
    Mohinh_KNN.fit(X_train, y_train)
    y_pred = Mohinh_KNN.predict(X_test)
    print("Accuracy is KNN: ",accuracy_score(y_test,y_pred)*100)
    arrKNN.append(accuracy_score(y_test,y_pred)*100)
    model=GaussianNB()
    model.fit(X_train, y_train)
    bayes_dubao = model.predict(X_test)
    print("Accuracy is Bayes: ",accuracy_score(y_test,bayes_dubao)*100)
    arrBayes.append(accuracy_score(y_test,bayes_dubao)*100)
    Cay_quyet_dinh = DecisionTreeClassifier(criterion="gini",random_state=100,max_depth=7,min_samples_leaf=5);
    Cay_quyet_dinh.fit(X_train,y_train)
    Cay_quyet_dinh_pred = Cay_quyet_dinh.predict(X_test)
    print("Accuracy is DecisionTree: ",accuracy_score(y_test,Cay_quyet_dinh_pred)*100)
    arrTree.append(accuracy_score(y_test,Cay_quyet_dinh_pred)*100)

s = np.arange(10)
width = 0.2
plt.bar(s-width,arrKNN,width,label="KNN")
plt.bar(s,arrBayes,width,label="Bayes")
plt.bar(s+width,arrTree,width,label="Tree")
plt.show()
