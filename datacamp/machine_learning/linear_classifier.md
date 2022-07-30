# Linear Classifier in Python

Content:
- Chapter 1: Applying logistic regression and SVM
- Chapter 2: Loss functions
- Chapter 3: Logistic regression
- Chapter 4: Support Vector Machines

### Chapter 1: Applying logistic regression and SVM
---
Code example of Logistic Regression:

```python
import sklearn.datasets
wine = sklearn.datasets.load_wine()

from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(wine.data, wine.target)
# Score
lr.score(wine.data, wine.target)
# Probability 
lr.predict_proba(wine.data[:1])

```
Code example of SVC:
```python
import sklearn.datasets

wine = sklearn.datasets.load_wine()
from sklearn.svm import LinearSVC

svm = LinearSVC()

svm.fit(wine.data, wine.target)
svm.score(wine.data, wine.target)
```

> Notes:
> 
> **Overfitting**: Training Score > Test Score
> 
> **Underfitting**: Training Score < Test Score

### Chapter 2: Loss functions
---
