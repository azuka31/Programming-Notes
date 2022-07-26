## Unsupervised Learning in Python
Cluster, transform, visualize, and extract insights from unlabeled datasets using scikit-learn and scipy

---

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
