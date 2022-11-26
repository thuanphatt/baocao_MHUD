import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix
import matplotlib.pyplot as plt

# Đọc dữ liệu

data=pd.read_csv("full_data.csv" )
x = data.iloc[:,0:10]
y = data.stroke
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3.0, random_state=100)

# Mô hình KNN
#dieu chinh k lang gieng gan nhat tu tap huan luyen 
Mohinh_KNN = KNeighborsClassifier(n_neighbors=13)
Mohinh_KNN.fit(x_train, y_train)
y_pred_KNN = Mohinh_KNN.predict(x_test)
print("Accuracy KNN is: ", accuracy_score(y_test, y_pred_KNN)*100) # 95.305%
print("F1 score KNN is",f1_score(y_test, y_pred_KNN)); 
cnf_matrix_gnb = confusion_matrix(y_test,y_pred_KNN,labels=[1,0]) #Nhãn của Stroke
print(cnf_matrix_gnb)

# Bayes thơ ngây

model = GaussianNB()
model.fit(x_train, y_train)
y_pred_Bayes = model.predict(x_test) # Dự đoán nhãn cho các phần tử trong tập kiểm tra
print("Accuracy Bayes is: ", accuracy_score(y_test, y_pred_Bayes)*100) # 86.326%
print("F1 score Bayes is",f1_score(y_test, y_pred_Bayes));

# Cây quyết định

cayquyetdinh = DecisionTreeClassifier(criterion = 'gini', random_state = 100, max_depth = 7, min_samples_leaf = 5)
# Xây dựng mô hình cây quyết định dựa trên chỉ số Gini với độ sâu của cây bằng 7, nút nhánh ít nhất có 5 phần tử
cayquyetdinh.fit(x_train,y_train)
y_pred_Tree = cayquyetdinh.predict(x_test)
print('Accuracy DecisionTree is: ', accuracy_score(y_test, y_pred_Tree)*100) # 94.953%
print("F1 score DecisionTree is",f1_score(y_test, y_pred_Tree));
print("=============== Độ chính xác và đánh giá chất lượng của mô hình(F1) qua các lần lặp với k láng giềng khác nhau ===============");
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=1/3.0, random_state=100)
acc=0
n=0
for k in [3,5,7,9,11,13,15,17,19]:
    Mohinh_KNN = KNeighborsClassifier(n_neighbors=k)
    Mohinh_KNN.fit(x_train, y_train)
    y_pred_KNN = Mohinh_KNN.predict(x_test)
    print("Giá trị K: ", k)
    print("Accuracy KNN is: ", accuracy_score(y_test, y_pred_KNN)*100)
    print("F1 score KNN is",f1_score(y_test, y_pred_KNN));
    if(acc<=accuracy_score(y_test, y_pred_KNN)*100):
        acc=accuracy_score(y_test, y_pred_KNN)*100
        n=k
print("================================")
print("Accuracy max: ",acc)
print("Giá trị k có độ chính xác cao nhất: ",n) 
print("================================")



# Phần vẽ sơ đồ

arrKNN = []
arrBayes = []
arrTree = []
F1KNNis = []
F1Bayesis = []
F1Treeis = []

for i in range(0,10):
    X_train,X_test,y_train, y_test=train_test_split(x, y, test_size=1/3.0,random_state=100+i)
    Mohinh_KNN = KNeighborsClassifier(n_neighbors=3)
    Mohinh_KNN.fit(X_train, y_train)
    y_pred = Mohinh_KNN.predict(X_test)
    print("Accuracy KNN is: ",accuracy_score(y_test,y_pred)*100, i)
    print("F1 score KNN is",f1_score(y_test, y_pred_KNN));
    arrKNN.append(accuracy_score(y_test,y_pred)*100)
    F1KNNis.append(f1_score(y_test, y_pred_KNN))
    model=GaussianNB()
    model.fit(X_train, y_train)
    bayes_dubao = model.predict(X_test)
    print("Accuracy Bayes is: ",accuracy_score(y_test,bayes_dubao)*100)
    print("F1 score Bayes is",f1_score(y_test, bayes_dubao));
    arrBayes.append(accuracy_score(y_test,bayes_dubao)*100)
    F1Bayesis.append(f1_score(y_test, bayes_dubao))
    Cay_quyet_dinh = DecisionTreeClassifier(criterion="gini",random_state=100,max_depth=7,min_samples_leaf=5);
    Cay_quyet_dinh.fit(X_train,y_train)
    Cay_quyet_dinh_pred = Cay_quyet_dinh.predict(X_test)
    print("Accuracy DecisionTree is : ",accuracy_score(y_test,Cay_quyet_dinh_pred)*100)
    print("F1 score DecisionTree is",f1_score(y_test, Cay_quyet_dinh_pred));
    arrTree.append(accuracy_score(y_test,Cay_quyet_dinh_pred)*100)
    F1Treeis.append(f1_score(y_test, Cay_quyet_dinh_pred))

s = np.arange(10)
width = 0.2
plt.bar(s-width,arrKNN,width,label="KNN")
plt.bar(s,arrBayes,width,label="Bayes")
plt.bar(s+width,arrTree,width,label="Tree")
plt.show()

plt.bar(s+width,F1KNNis,width,label="F1 Score KNN")
plt.bar(s,F1Bayesis,width,label="F1 Score Bayes")
plt.bar(s-width,F1Treeis,width,label="F1 Score DecisionTree")
plt.show()
