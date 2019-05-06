
# Performing Principal Component Analysis (PCA)

## Introduction

In this lesson, you'll code PCA from the ground up using NumPy. This should provide you with a deeper understanding of the algorithm and continue to practice your linear algebra skills.

## Objectives

You will be able to:

- Understand the steps required to perform PCA on a given dataset
- Understand and explain the role of Eigen decomposition in PCA


## Step 1: Get some data

To start, generate some data for PCA!


```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

x1 = np.random.uniform(low=0, high=10, size=100)
x2 = [(xi*3)+np.random.normal(scale=2) for xi in x1]
plt.scatter(x1,x2);
```


![png](index_files/index_2_0.png)


## Step 2: Subtract the mean

Next, you have to subtract the mean from each dimension of the data. So, all the $x$ values
have $\bar{x}$ (the mean of the $x$ values of all the data points) subtracted, and all the $y$ values
have $\bar{y}$ subtracted from them. 


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
      <td>1.032030</td>
      <td>-0.492450</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3.979683</td>
      <td>9.763812</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8.883582</td>
      <td>28.979365</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9.893061</td>
      <td>28.469174</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8.145913</td>
      <td>25.109462</td>
    </tr>
  </tbody>
</table>
</div>




```python
data.mean()
```




    x1     4.535001
    x2    13.566201
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
      <td>-3.502971</td>
      <td>-14.058652</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-0.555318</td>
      <td>-3.802390</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.348581</td>
      <td>15.413164</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5.358060</td>
      <td>14.902973</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.610912</td>
      <td>11.543261</td>
    </tr>
  </tbody>
</table>
</div>



## Step 3: Calculate the covariance matrix

Now that you have normalized your data, you must now calculate the covariance matrix.


```python
cov = np.cov([mean_centered.x1, mean_centered.x2])
cov
```




    array([[ 7.15958302, 21.47416477],
           [21.47416477, 67.92443943]])



## Step 4: Calculate the eigenvectors and eigenvalues of the covariance matrix

Now that you've calculated the covariance matrix, its time to compute the associated eigenvectors. These will form the new axes when its time to reproject the dataset on the new basis.


```python
eigen_value, eigen_vector = np.linalg.eig(cov)
eigen_vector
```




    array([[-0.95305204, -0.30280656],
           [ 0.30280656, -0.95305204]])




```python
eigen_value
```




    array([ 0.33674687, 74.74727559])



## Step 5: Choosing components and forming a feature vector

If you look at the eigenvectors and eigenvalues above, you can see that the eigenvalues have very different values. In fact, it turns out that **the eigenvector with the highest eigenvalue is the principal component of the data set.**


In general, once eigenvectors are found from the covariance matrix, the next step is to order them by eigenvalue, highest to lowest. This gives us the components in order of significance. Typically, PCA will be used to reduce the dimensionality of the dataset and, as such, some of these eigenvectors will be subsequently discarded. In general, the smaller the eigenvalue relative to others, the less information encoded within said feature.

Finally, you need to form a __feature vector__, which is just a fancy name for a matrix of vectors. This is constructed by taking the eigenvectors that you want to keep from the list of eigenvectors, and forming a matrix with these eigenvectors in the columns as shown below:


```python
e_indices = np.argsort(eigen_value)[::-1] #Get the index values of the sorted eigenvalues
eigenvectors_sorted = eigen_vector[:,e_indices]
eigenvectors_sorted
```




    array([[-0.30280656, -0.95305204],
           [-0.95305204,  0.30280656]])



## Step 5: Deriving the new data set

This the final step in PCA, and is also the easiest. Once you have chosen the components (eigenvectors) that you wish to keep in our data and formed a feature vector, you simply take the transpose of the vector and multiply it on the left of the original data set, transposed.


```python
transformed = eigenvectors_sorted.dot(mean_centered.T).T
transformed[:5]
```




    array([[ 14.4593493 ,  -0.91853819],
           [  3.79202935,  -0.62214167],
           [-16.00632588,   0.52278351],
           [-15.82576429,  -0.59379253],
           [-12.0947357 ,   0.05398833]])



## Summary 

That's it! You just coded PCA on your own using NumPy! In the next lab, you'll continue to practice this on your own!
