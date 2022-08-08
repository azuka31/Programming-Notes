# Preprocessing for Machine Learning in Python
Content:
- Introduction to Data Preprocessing
- Standardizing Data
- Feature Engineering
- Selecting features for modeling
- Putting it all together

### Chapter 1. Introduction to Data Preprocessing
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

### Chapter 2. Standardizing Data
---
Log Normalization
```python
# Print out the variance of the Proline column
print(wine['Proline'].var())

# Apply the log normalization function to the Proline column
wine['Proline_log'] = np.log(wine['Proline'])

# Check the variance of the normalized Proline column
print(wine['Proline_log'].var())
```
```
99166.71735542436
0.17231366191842012
```
Standard Scaling Example
```python
# Import StandardScaler from scikit-learn
from sklearn.preprocessing import StandardScaler

# Create the scaler
ss = StandardScaler()

# Take a subset of the DataFrame you want to scale 
wine_subset = wine[['Ash', 'Alcalinity of ash', 'Magnesium']]

# Apply the scaler to the DataFrame subset
wine_subset_scaled = ss.fit_transform(wine_subset)
```
### Chapter 3. Feature Engineering
---
Example
- Text Vectorizer
- Categorical -> Numerical
- Time series

Encoding Categorical Variables
LabelEncoder Example:
```python
# Set up the LabelEncoder object
enc = LabelEncoder()

# Apply the encoding to the "Accessible" column
hiking['Accessible_enc'] = enc.fit_transform(hiking['Accessible'])

# Compare the two columns
print(hiking[['Accessible', 'Accessible_enc']].head())
```
```
      Accessible  Accessible_enc
    0          Y               1
    1          N               0
    2          N               0
    3          N               0
    4          N               0
```
One Hot Encoder Example:
```python
# Transform the category_desc column
category_enc = pd.get_dummies(volunteer["category_desc"])
```

Datetime Problem
Converting Datetime -> year + month + date

Text Problem Approaches
- regular expression
- vectorizing text (tf-idf)

Regex Examples:
```python
# Write a pattern to extract numbers and decimals
def return_mileage(length):
    pattern = re.compile(r"\d+\.\d+")
    
    # Search the text for matches
    mile = re.match(pattern, length)
    
    # If a value is returned, use group(0) to return the found value
    if mile is not None:
        return float(mile.group(0))
        
# Apply the function to the Length column and take a look at both columns
hiking["Length_num"] = hiking['Length'].apply(lambda row: return_mileage(row))
print(hiking[["Length", "Length_num"]].head())
```
```
<script.py> output:
           Length  Length_num
    0   0.8 miles        0.80
    1    1.0 mile        1.00
    2  0.75 miles        0.75
    3   0.5 miles        0.50
    4   0.5 miles        0.50
```

TFIDF examples:
```python
# Take the title text
title_text = volunteer['title']

# Create the vectorizer method
tfidf_vec = TfidfVectorizer()

# Transform the text into tf-idf vectors
text_tfidf = tfidf_vec.fit_transform(title_text)

# Split the dataset according to the class distribution of category_desc
y = volunteer["category_desc"]
X_train, X_test, y_train, y_test = train_test_split(text_tfidf.toarray(), y, stratify=y)

# Fit the model to the training data
nb.fit(X_train, y_train)

# Print out the model's accuracy
print(nb.score(X_test, y_test))
```
