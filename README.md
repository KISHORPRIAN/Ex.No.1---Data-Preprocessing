# Ex.No.1---Data-Preprocessing
## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

##REQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

Kaggle :
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

Data Preprocessing:

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

Need of Data Preprocessing :

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
Importing the libraries
Importing the dataset
Taking care of missing data
Encoding categorical data
Normalizing the data
Splitting the data into test and train

## PROGRAM:
from google.colab import files
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
df=pd.read_csv('/content/data.csv')
print(df)
X=df.iloc[:,:-1].values
print(X)
Y=df.iloc[:,-1].values
print(Y)
print(df.isnull().sum())
df.fillna(df.mean().round(1),inplace=True)
print(df.isnull().sum())
Y= df.iloc[:,-1].values
print(Y)
df.duplicated()
print(df['Calories'].describe())
scaler= MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(df))
print(df1)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2)
print(X_train)
print(len(X_train))
print(X_test)
print(len(X_test))


## OUTPUT:
![image](https://github.com/gokulvijayaramanuja/Ex.No.1---Data-Preprocessing/assets/119577543/c5f747c3-1f52-4d7a-9958-f8ac7e8503ae)

![image](https://github.com/gokulvijayaramanuja/Ex.No.1---Data-Preprocessing/assets/119577543/4e62d048-21c2-4b16-a945-4b1ea1574cea)

![image](https://github.com/gokulvijayaramanuja/Ex.No.1---Data-Preprocessing/assets/119577543/6a1624a9-9fd5-4abf-b01e-58e63cb27c2f)

![image](https://github.com/gokulvijayaramanuja/Ex.No.1---Data-Preprocessing/assets/119577543/f41e1030-4d41-4248-9954-9c312847333b)

![image](https://github.com/gokulvijayaramanuja/Ex.No.1---Data-Preprocessing/assets/119577543/7922da4d-0eaa-4dbc-b287-37f5fdbfb213)

![image](https://github.com/gokulvijayaramanuja/Ex.No.1---Data-Preprocessing/assets/119577543/d0e1059a-2c68-4d6f-9356-bf0b0bc2814c)

![image](https://github.com/gokulvijayaramanuja/Ex.No.1---Data-Preprocessing/assets/119577543/99804a48-708d-49c2-a3fe-d3de30505ef9)

![image](https://github.com/gokulvijayaramanuja/Ex.No.1---Data-Preprocessing/assets/119577543/d440abca-ef9c-466d-9ab8-581710065408)
## RESULT
The program executed successfully
