import numpy as np 
import pandas as pd 

import seaborn as sns
sns.set_style("whitegrid")
from matplotlib import pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
scaler = MinMaxScaler()

from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBClassifier

training = pd.read_csv("train.csv")
testing = pd.read_csv("test.csv")

#sibsp(sibling=kardeş ve spouse=eş)
#parch(parent=ebeveyn ve child=cocuk)

print(training.head())
print(testing.head())
print(training.keys())
print(testing.keys())
print(training.describe())

print("Training Kayıp Veri Sayısı")
print(pd.isnull(training).sum())
print("Testing Kayıp Veri Sayısı")
print(pd.isnull(testing).sum())

print(training.shape)
print(training.info())
print(training.describe())

sns.heatmap(training.isnull(),yticklabels=False,cbar=False)
plt.show()

training.drop(labels = ["Cabin", "Ticket"], axis = 1, inplace = True)
testing.drop(labels = ["Cabin", "Ticket"], axis = 1, inplace = True)

#Boşları doldurma
training["Age"].fillna(training["Age"].median(), inplace = True)
testing["Age"].fillna(testing["Age"].median(), inplace = True) 
training["Embarked"].fillna("S", inplace = True)
testing["Fare"].fillna(testing["Fare"].median(), inplace = True)

#Cinsiyet(Sex)
f,ax=plt.subplots(1,2,figsize=(18,8))
training['Survived'].value_counts().plot.pie(explode=[0,0.1],autopct='%1.1f%%',ax=ax[0],shadow=True)
ax[0].set_title('Survived')
ax[0].set_ylabel('')
sns.countplot('Sex',hue='Survived',data=training,ax=ax[1])
ax[1].set_title('Survived')
plt.title("Cinsiyete Göre Hayatta Kalma Dağılımı")
plt.show()
toplam_yasayan_kadın = training[training.Sex == "female"]["Survived"].sum()
toplam_yasayan_erkek = training[training.Sex == "male"]["Survived"].sum()
print("Toplam Hayatta Kalan İnsan: " + str((toplam_yasayan_kadın + toplam_yasayan_erkek)))
print("Hayatta Kalan Kadın Oranı:") 
print(toplam_yasayan_kadın/(toplam_yasayan_kadın + toplam_yasayan_erkek))
print("Hayatta Kalan Erkek Oranı:")
print(toplam_yasayan_erkek/(toplam_yasayan_kadın + toplam_yasayan_erkek))

#Sınıf(Pclass)
sns.countplot('Pclass', hue='Survived', data=training)
plt.title("Sınıfa Göre Hayatta Kalma Dağılımı")
plt.show()
toplam_yasayan_bir = training[training.Pclass == 1]["Survived"].sum()
toplam_yasayan_iki = training[training.Pclass == 2]["Survived"].sum()
toplam_yasayan_uc = training[training.Pclass == 3]["Survived"].sum()
toplam_yasayan_sinif = toplam_yasayan_bir + toplam_yasayan_iki + toplam_yasayan_uc
print("Toplam Hayatta Kalan İnsan: " + str(toplam_yasayan_sinif))
print("1.Sınıftaki Yolcuların Hayatta Kalma Oranı:") 
print(toplam_yasayan_bir/toplam_yasayan_sinif)
print("2.Sınıftaki Yolcuların Hayatta Kalma Oranı:")
print(toplam_yasayan_iki/toplam_yasayan_sinif)
print("3.Sınıftaki Yolcuların Hayatta Kalma Oranı:")
print(toplam_yasayan_uc/toplam_yasayan_sinif)
sns.countplot("Pclass", hue="Sex", data=training)
plt.title("Cinsiyet ve Sınıfa Göre Yaşama Oranları")
plt.show()

#Yaş(Age)
yasayan_ages = training[training.Survived == 1]["Age"]
yasamayan_ages = training[training.Survived == 0]["Age"]
plt.subplot(1, 2, 1)
sns.distplot(yasayan_ages, kde=False)
plt.axis([0, 100, 0, 100])
plt.title("Hayatta Kalanlar")
plt.subplot(1, 2, 2)
sns.distplot(yasamayan_ages, kde=False)
plt.axis([0, 100, 0, 100])
plt.title("Hayatta Kalamayanlar")
plt.subplots_adjust()
plt.show()

#Scatterplot Pclass, Age, Fare, Survived
g = sns.FacetGrid(training, hue="Survived", col="Pclass", margin_titles=True,
                  palette={1:"green", 0:"black"})
g=g.map(plt.scatter, "Fare", "Age",edgecolor="w").add_legend()
training.plot(kind='scatter', x='Age', y='Fare',alpha = 0.5,color = 'red')
plt.show()

#Box Age, Pclass
ax= sns.boxplot(x="Pclass", y="Age", data=training)
ax= sns.stripplot(x="Pclass", y="Age", data=training, jitter=True, edgecolor="gray")
plt.show()

#Histogram
training.hist(figsize=(15,20))
plt.show()

#kdeplot
sns.FacetGrid(training, hue="Survived", height=5).map(sns.kdeplot, "Fare").add_legend()
plt.show()

#jointplot Fare, Age
sns.jointplot(x='Fare',y='Age',data=training)
plt.show()

#Swarmplot
sns.swarmplot(x='Pclass',y='Age',data=training)
plt.show()

#Catplot
sns.catplot('Pclass','Survived',hue='Sex',data=training, kind="point")
plt.show()
f,ax=plt.subplots(1,2,figsize=(20,8))
sns.barplot('SibSp','Survived', data=training, ax=ax[0])
sns.catplot('SibSp','Survived', data=training, kind="point", ax=ax[1])
plt.close(2)
plt.show()



#Sex ve Embarked sütunlarını sayısal değere çevirme
label_sex = LabelEncoder()
label_sex.fit(training["Sex"])

encoded_sex_training = label_sex.transform(training["Sex"])
training["Sex"] = encoded_sex_training
encoded_sex_testing = label_sex.transform(testing["Sex"])
testing["Sex"] = encoded_sex_testing

label_embarked = LabelEncoder()
label_embarked.fit(training["Embarked"])

encoded_embarked_training = label_embarked.transform(training["Embarked"])
training["Embarked"] = encoded_embarked_training
encoded_embarked_testing = label_embarked.transform(testing["Embarked"])
testing["Embarked"] = encoded_embarked_testing

print(testing.sample(5))
print(training.sample(5))

#Yeni sütunlar
training["FamilySize"] = training["SibSp"] + training["Parch"] + 1
testing["FamilySize"] = testing["SibSp"] + testing["Parch"] + 1
training["IsAlone"] = training.FamilySize.apply(lambda x: 1 if x == 1 else 0)
testing["IsAlone"] = testing.FamilySize.apply(lambda x: 1 if x == 1 else 0)

for name in training["Name"]:
    training["Title"] = training["Name"].str.extract("([A-Za-z]+)\.",expand=True)
    
for name in testing["Name"]:
    testing["Title"] = testing["Name"].str.extract("([A-Za-z]+)\.",expand=True)

titles = set(training["Title"])
print(titles)

#title sayıları
title_list = list(training["Title"])
frequency_titles = []

for i in titles:
    frequency_titles.append(title_list.count(i))
    
print(frequency_titles)

titles = list(titles)

title_dataframe = pd.DataFrame({
    "Titles" : titles,
    "Frequency" : frequency_titles
})

print(title_dataframe)

#küçültme ve sayısallaştırma
title_replacements = {"Mlle": "Other", "Major": "Other", "Col": "Other", "Sir": "Other", "Don": "Other", "Mme": "Other",
          "Jonkheer": "Other", "Lady": "Other", "Capt": "Other", "Countess": "Other", "Ms": "Other", "Dona": "Other"}

training.replace({"Title": title_replacements}, inplace=True)
testing.replace({"Title": title_replacements}, inplace=True)

le_title = LabelEncoder()
le_title.fit(training["Title"])

encoded_title_training = le_title.transform(training["Title"])
training["Title"] = encoded_title_training
encoded_title_testing = le_title.transform(testing["Title"])
testing["Title"] = encoded_title_testing

#name sütunu çıkarma
training.drop("Name", axis = 1, inplace = True)
testing.drop("Name", axis = 1, inplace = True)
print(training.sample(5))
print(testing.sample(5))

#Verileri tekrar şekillendiriyoruz
ages_train = np.array(training["Age"]).reshape(-1, 1)
fares_train = np.array(training["Fare"]).reshape(-1, 1)
ages_test = np.array(testing["Age"]).reshape(-1, 1)
fares_test = np.array(testing["Fare"]).reshape(-1, 1)

training["Age"] = scaler.fit_transform(ages_train)
training["Fare"] = scaler.fit_transform(fares_train)
testing["Age"] = scaler.fit_transform(ages_test)
testing["Fare"] = scaler.fit_transform(fares_test)

print(training.head())
print(testing.head())

#MODELLER
X_train = training.drop(labels=["PassengerId", "Survived"], axis=1)
y_train = training["Survived"]

x_train, x_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

# Gaussian Naive Bayes
gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
y_pred = gaussian.predict(x_val)
acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)
print("Gaussian Naive Bayes = " + str(acc_gaussian))

# Logistic Regression
logreg = LogisticRegression(solver='lbfgs')
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_val)
acc_logreg = round(accuracy_score(y_pred, y_val) * 100, 2)
print("Logistic Regression = " + str(acc_logreg))

#XGBoost Classifier
xgboost = XGBClassifier(max_depth=5, n_estimators=500, learning_rate=0.01)
xgboost.fit(x_train, y_train)
y_pred = xgboost.predict(x_val)
acc_xgboost = round(accuracy_score(y_pred, y_val) * 100, 2)
print("XGBoostClassifier = " + str(acc_xgboost))


# Support Vector Machines
svc = SVC(gamma='scale')
svc.fit(x_train, y_train)
y_pred = svc.predict(x_val)
acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print("Support Vector Machines = " + str(acc_svc))

# Linear SVC
linear_svc = LinearSVC(max_iter=5000)
linear_svc.fit(x_train, y_train)
y_pred = linear_svc.predict(x_val)
acc_linear_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
print("Linear SVC = " + str(acc_linear_svc))

# Perceptron
perceptron = Perceptron()
perceptron.fit(x_train, y_train)
y_pred = perceptron.predict(x_val)
acc_perceptron = round(accuracy_score(y_pred, y_val) * 100, 2)
print("Perceptron = " + str(acc_perceptron))

#Decision Tree
decisiontree = DecisionTreeClassifier()
decisiontree.fit(x_train, y_train)
y_pred = decisiontree.predict(x_val)
acc_decisiontree = round(accuracy_score(y_pred, y_val) * 100, 2)
print("Decision Tree = " + str(acc_decisiontree))

# Random Forest
randomforest = RandomForestClassifier(n_estimators=100)
randomforest.fit(x_train, y_train)
y_pred = randomforest.predict(x_val)
acc_randomforest = round(accuracy_score(y_pred, y_val) * 100, 2)
print("Random Forest = " + str(acc_randomforest))

# KNN or k-Nearest Neighbors
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
y_pred = knn.predict(x_val)
acc_knn = round(accuracy_score(y_pred, y_val) * 100, 2)
print("KNN = " + str(acc_knn))

# Stochastic Gradient Descent
sgd = SGDClassifier()
sgd.fit(x_train, y_train)
y_pred = sgd.predict(x_val)
acc_sgd = round(accuracy_score(y_pred, y_val) * 100, 2)
print("Stochastic Gradient Descent = " + str(acc_sgd))

# Gradient Boosting Classifier
gbc = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', 
                                min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, 
                                min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=None, max_features=None, 
                                verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto', validation_fraction=0.1, 
                                n_iter_no_change=None, tol=0.0001)
gbc.fit(x_train, y_train)
y_pred = gbc.predict(x_val)
acc_gbc = round(accuracy_score(y_pred, y_val) * 100, 2)
score = mean_absolute_error(y_val, y_pred)
print("Gradient Boosting Classifier = " + str(acc_gbc))
print("Skor = " + str(score))

models = pd.DataFrame({
    'Model': ['Support Vector Machines', 'KNN', 'Logistic Regression', 'XGBoost', 
              'Random Forest', 'Naive Bayes', 'Perceptron', 'Linear SVC', 
              'Decision Tree', 'Stochastic Gradient Descent', 'Gradient Boosting Classifier'],
    'Score': [acc_svc, acc_knn, acc_logreg, acc_xgboost,
              acc_randomforest, acc_gaussian, acc_perceptron, acc_linear_svc, acc_decisiontree,
              acc_sgd, acc_gbc]})
print(models.sort_values(by='Score', ascending=False))

#Dosya oluşturma
ids = testing['PassengerId']
predictions = gbc.predict(testing.drop('PassengerId', axis=1))
output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': predictions })
output.to_csv('sonuclar.csv', index=False)
