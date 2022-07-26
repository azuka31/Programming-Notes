# Machine Learning Track

## Unsupervised Learning in Python
> Cluster, transform, visualize, and extract insights from unlabeled datasets using scikit-learn and scipy

Chapter Overview:
- Chapter 1: Clustering for dataset exploration
- Chapter 2: Visualizatin with hierarchical clustering and t-SNE
- Chapter 3: Decorrelating your data and dimesion reduction
- Chapter 4: Discovering interpretable features

### Chapter 1. Clustering for dataset exploration
---
> Practicing Clustering using sklearn
```python
# Sample Data 
sample = [[ 5. 3.3 1.4 0.2]
[ 5. 3.5 1.3 0.3]
...
[ 7.2 3.2 6. 1.8]]

# Executing Scikit Learn
from sklearn.cluster import KMeans
model = KMeans(n_clusters=3)
model.fit(samples)
lables = model.predict(samples)

# Visualizing
import matplotlib.pyplot as plt
xs = samples[:,0]
ys = samples[:,2]
plt.scatter(xs, ys, c=labels)
plt.show()
```
![output1](output/ouput1.png)

> Evaluating Cluster
```python
import pandas as pd
df = pd.DataFrame({'labels': labels, 'species': species})
ct = pd.crosstab(df['labels'], df['species'])
print(ct)
```
```python
# Measuring Inertia
from sklearn.cluster import KMeans

model = KMeans(n_clusters=3)

model.fit(samples)
print(model.inertia_)
```
> you can loop to get other intertia to observe best clustering based on 'elbow law'

![output2](output/output2.png)

> Standar Scaler
> 
```python
# to create standarization, mean 0 and variance 1
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(samples)
StandardScaler(copy=True, with_mean=True, with_std=True)

samples_scaled = scaler.transform(samples)

# Or using Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
scaler = StandardScaler()
kmeans = KMeans(n_clusters=3)

from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(scaler, kmeans)
pipeline.fit(samples)
```
> Note:
>
> fit()/predict() in model, fit()/transform() in scaler
>
> Here is [youtube](https://www.youtube.com/watch?v=2tuBREK_mgE) reference to understand standarization mean

---

## Linear Classifier in Python
> The details of linear classifiers like logistic and SVM

Chapter Overview:
- Chapter 1: Applying logistic regression and SVM
- Chapter 2: Loss functions
- Chapter 3: Logistic regression
- Chapter 4: Support Vector Machines

---
## Machine Learning with Tree-Based Models in Python
> Tree-based models and ensembles for regression and classification using scikit-learn

Chapter Overview:
- Chapter 1: Classification and Regression Trees
- Chapter 2: The Bias-Variance Tradeoff
- Chapter 3: Bagging and Random Forests
- Chapter 4: Boosting
- Chapter 5: Model Tuning

---
## Extreme Gradient Boosting with XGBoost
> Gradient Boosting and build state-of-the-art machine learning models using XGBoost

Chapter Overview:
- Chapter 1: Classification with XGBoost
- Chapter 2: Regression with XGBoost
- Chapter 3: Fine-tuning your XGBoost model
- Chapter 4: Using XGBoost in pipelines

---
## Cluster Analysis in Python
> Deep dive about hierarchical and k-means clusttering method

Chapter Overview:
- Chapter 1: Introduction to Clustering
- Chapter 2: Hierarchical Clustering
- Chapter 3: K-Means Clustering
- Chapter 4: Clustering in Real World

---
## Dimensionality Reduction in Python
> On progress, reducing dimensionality basic concept
