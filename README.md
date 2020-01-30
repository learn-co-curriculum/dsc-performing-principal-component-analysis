# Performing Principal Component Analysis (PCA)

## Introduction

In this lesson, we'll write the PCA algorithm from the ground up using NumPy. This should provide you with a deeper understanding of the algorithm and help you practice your linear algebra skills.

## Objectives

You will be able to:

- List the steps required to perform PCA on a given dataset 
- Decompose and reconstruct a matrix using eigendecomposition 
- Perform a covariance matrix calculation with NumPy 


## Step 1: Get some data

To start, generate some data for PCA!


```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
np.random.seed(42)

x1 = np.random.uniform(low=0, high=10, size=100)
x2 = [(xi*3)+np.random.normal(scale=2) for xi in x1]
plt.scatter(x1, x2);
```


![png](index_files/index_2_0.png)


## Step 2: Subtract the mean

Next, you have to subtract the mean from each dimension of the data. 


```python
import pandas as pd

data = pd.DataFrame([x1,x2]).transpose()
data.columns = ['x1', 'x2']
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x1</th>
      <th>x2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.745401</td>
      <td>11.410298</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9.507143</td>
      <td>27.923414</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.319939</td>
      <td>22.143340</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5.986585</td>
      <td>13.984617</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.560186</td>
      <td>4.241215</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.mean()
```




    x1     4.701807
    x2    14.103262
    dtype: float64




```python
mean_centered = data - data.mean()
mean_centered.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>x1</th>
      <th>x2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-0.956406</td>
      <td>-2.692964</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.805336</td>
      <td>13.820153</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.618132</td>
      <td>8.040078</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.284777</td>
      <td>-0.118645</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-3.141621</td>
      <td>-9.862046</td>
    </tr>
  </tbody>
</table>
</div>



## Step 3: Calculate the covariance matrix

Now that you have normalized your data, you can calculate the covariance matrix.


```python
cov = np.cov([mean_centered['x1'], mean_centered['x2']])
cov
```




    array([[ 8.84999497, 25.73618675],
           [25.73618675, 78.10092586]])



## Step 4: Calculate the eigenvectors and eigenvalues of the covariance matrix

It's time to compute the associated eigenvectors. These will form the new axes when it's time to reproject the dataset on the new basis.


```python
eigen_value, eigen_vector = np.linalg.eig(cov)
eigen_vector
```




    array([[-0.94936397, -0.31417837],
           [ 0.31417837, -0.94936397]])




```python
eigen_value
```




    array([ 0.33297363, 86.61794719])



## Step 5: Choosing components and forming a feature vector

If you look at the eigenvectors and eigenvalues above, you can see that the eigenvalues have very different values. In fact, it turns out that **the eigenvector with the highest eigenvalue is the principal component of the dataset.**

In general, once eigenvectors are found from the covariance matrix, the next step is to order them by eigenvalue in descending order. This gives us the components in order of significance. Typically, PCA will be used to reduce the dimensionality of the dataset and as such, some of these eigenvectors will be subsequently discarded. In general, the smaller the eigenvalue relative to others, the less information encoded within said feature.

Finally, you need to form a __feature vector__, which is just a fancy name for a matrix of vectors. This is constructed by taking the eigenvectors that you want to keep from the list of eigenvectors, and forming a matrix with these eigenvectors in the columns as shown below:


```python
# Get the index values of the sorted eigenvalues
e_indices = np.argsort(eigen_value)[::-1] 

# Sort
eigenvectors_sorted = eigen_vector[:, e_indices]
eigenvectors_sorted
```




    array([[-0.31417837, -0.94936397],
           [-0.94936397,  0.31417837]])



## Step 5: Deriving the new dataset

This the final step in PCA, and is also the easiest. Once you have chosen the components (eigenvectors) that you wish to keep in our data and formed a feature vector, you simply take the transpose of the vector and multiply it on the left of the original dataset, transposed.


```python
transformed = eigenvectors_sorted.dot(mean_centered.T).T
transformed[:5]
```




    array([[  2.85708504,   0.06190663],
           [-14.63008778,  -0.2200194 ],
           [ -8.45552104,   0.04045849],
           [ -0.29101209,  -1.25699704],
           [ 10.34970068,  -0.11589976]])



## Summary 

That's it! We just implemented PCA on our own using NumPy! In the next lab, you'll continue to practice this on your own!
