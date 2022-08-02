# Preprocessing for Machine Learning in Python
Content:
- Introduction to Data Preprocessing
- Standardizing Data
- Feature Engineering
- Selecting features for modeling
- Putting it all together

### Introduction to Data Preprocessing
---
Working with data types
Basic desc
```python
print(volunteer.dtypes)
```
Converting columns
```python
# First method
df['c'] = df['c'].astype(float)
```

Class Distribution Stratified Sampling in Train test
Checking
```python
# Create a data with all columns except category_desc
volunteer_X = volunteer.drop('category_desc', axis=1)

# Create a category_desc labels dataset
volunteer_y = volunteer[['category_desc']]

# Use stratified sampling to split up the dataset according to the volunteer_y dataset
X_train, X_test, y_train, y_test = train_test_split(volunteer_X, volunteer_y, stratify=volunteer_y)

# Print out the category_desc counts on the training y labels
print(y_train['category_desc'].value_counts())
```

### Standardizing Data
---
