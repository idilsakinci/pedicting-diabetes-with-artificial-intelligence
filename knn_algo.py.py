#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Outcome = 1 Diabet
# Outcome 0 Healthy

data = pd.read_csv("diabetes.csv")
data.head()


# In[2]:


diabetes_people = data[data.Outcome == 1]
healthy_people = data[data.Outcome == 0]

# example graphic for age and glucose
plt.scatter(healthy_people.Age, healthy_people.Glucose, color="green", label="healthy", alpha = 0.4)
plt.scatter(diabetes_people.Age, diabetes_people.Glucose, color="red", label="diabetes", alpha = 0.4)

plt.xlabel("Age")
plt.ylabel("Glucose")

plt.legend()
plt.show()


# In[3]:


# diabetes or healthy
y = data.Outcome.values

# remove outcome datas (dependent)
x_raw_data = data.drop(["Outcome"], axis = 1) 
# normalization. so that high values do not overwhelm low values
x = (x_raw_data - np.min(x_raw_data)) / (np.max(x_raw_data) - np.min(x_raw_data))

print("Row datas before normalization:\n")
print(x_raw_data.head())

print("\n\nDatas after normalization:\n")
print(x.head())


# In[4]:


# separate the data into data to train and data to test
# change the test_size and test
# for example: 
# when the test_size = 0.2 (%80 for train, %20 for test) the result is 0.78,
# when the test_size = 0.1 (%90 for train, %10 for test) the result is 0.83,
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 1)

# knn modal
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)
print("Accuracy result of test data for k=3 ", knn.score(x_test, y_test))


# In[5]:


# best k value review
count = 1
for k in range(1, 11):
    knn_new = KNeighborsClassifier(n_neighbors = k)
    knn_new.fit(x_train, y_train)
    print(count, " ", "Accuracy rate: %", knn_new.score(x_test, y_test) * 100)
    count += 1


# In[ ]:




