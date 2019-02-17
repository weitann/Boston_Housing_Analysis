# Boston Housing Analysis

This notebook analyzes the median boston housing prices (MEDV) and explores both KNN and Linear Regression methods.

### KNN
* Comparison between with normalization and without normalization
* Effect of number of neighbors (k parameter) on RMSE
* Effect of L (in computing distance) on RMSE
* Forward selection of features using OLS

### Linear regression
* Flexibility of model 
* Gradient descent
* Effect of normalization and learning rate on processing time, ability to converge and number of iterations to minimize cost function
* Ridge regularization

## Brief introduction on the data set

The following features are used:

| Features | Description |
| ---         |     ---     |
| CRIM   | per capita crime rate by town     |
| ZN     | proportion of residential land zoned for lots over 25,000 sq.ft.       |
| INDUS   | proportion of non-retail business acres per town     |
| CHAS     | Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)       |
| NOX   | nitric oxides concentration (parts per 10 million)     |
| RM     | average number of rooms per dwelling       |
| AGE   | proportion of owner-occupied units built prior to 1940    |
| DIS     | weighted distances to five Boston employment centres       |
| RAD   | index of accessibility to radial highways    |
| TAX     | full-value property-tax rate per $10,000      |
| PTRATIO   | pupil-teacher ratio by town |
| B     | 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town       |
| LSTAT     | % lower status of the population   |
| MEDV     | Median value of owner-occupied homes in $1000's      |


The spread of the housing prices is displayed in the histogram below. 

![spread](https://user-images.githubusercontent.com/42630569/52907233-31558880-3212-11e9-8b3d-132d2d8233d4.png)

## Part I : K Nearest Neighbors

In the first part of this notebook, KNN is coded from scratch, and the hyperparameters are tuned to observed their respective impacts on the RMSE of housing predictions.

We will observe the effect of overfitting with the KNN model (k=1), and perform cross validation to determine the optimal hyperparameters to use.

![overfit](https://user-images.githubusercontent.com/42630569/52907265-dbcdab80-3212-11e9-89f5-8055465fccd6.png)

### Baseline 

Taking the mean as a baseline, we obtain a point of reference for comparison. With a mean of 22.79, the training and testing RMSE are 9.39 and 8.37 respectively. Testing RMSE is significantly lower, but since the mean is not a good estimate, this does not bring much value.

![baseline](https://user-images.githubusercontent.com/42630569/52907271-20f1dd80-3213-11e9-93c0-5f2d6fd77ccf.png)

### KNN function

The following KNN function is built to run the predictions. If the training set is also used as the testing set, this function will select the next closest point for the predictions.

```
def nneighbor(test_data, train_data, features, L, K):

    all_predicted_prices = []

    train = np.array(train_data[features])
    
    for index, row in test_data.iterrows():
        test = np.array(row[features])
        dist = np.sum(abs(test - train)**L, axis=1) **(1/L) 
        sort_dist_idx = np.argsort(dist)

        # check if data sets are the same, i.e. comparing training to training
        if np.array_equal(test_data, train_data):
            k_idx_list = sort_dist_idx[1 : 1 + K]
            list_rows = train_data.iloc[np.r_[k_idx_list],:]
            predicted_price = round(np.mean(list_rows["MEDV"]), 2)
            
        else:
            k_idx_list = sort_dist_idx[0 : 0 + K]
            list_rows = train_data.iloc[np.r_[k_idx_list],:]
            predicted_price = round(np.mean(list_rows["MEDV"]), 2)
        
        all_predicted_prices.append(predicted_price)

    return all_predicted_prices
```

### Without normalization : 2 features + k=1 + L=2

Using only 2 features: CRIM and RM, with k=1, and L=2, the training and testing RMSE are 7.28 and 7.4 respectively. And the scatterplot is as follows. Overfitting is not significant, but to increase accuracy, normalization and more features will be necessary.

![img1](https://user-images.githubusercontent.com/42630569/52907288-8b0a8280-3213-11e9-9d36-23f861c38dd4.png)

### With normalization : 2 features + k=1 + L=2

With all hyperparameters kept constant, but features normalized, we obtained a training and testing RMSE score of 7.69 and 6.17 respectively. with normalization, features are brought to the same scale and weighed equally, hence improving prediction accuracy.

**Conclusion 1 : Normalizing features in KNN model increases accuracy**



### Effect of more features and different L values

With more features, the accuracy for KNN increases, though with increasing processing time. with more features added - RM, LSTAT, CRIM, NOX, PTRATIO - the training and testing accuracy decreases to 5.39 and 4.08 respectively. 


| KNN model | Normalized | Test RMSE |
| ---         |     ---     |   ---     |
| 2 features, k=1, L=2   | No |7.4    |
| 2 features, k=1, L=2   | Yes  | 6.17       |
| 5 features, k=1, L=2  | Yes | 5.39     |
| 5 features, k=1, L=1    | Yes | 5.10       |
| 5 features, k=1, L=3   | Yes| 5.28     |



### Effect of varying k value

Using the following function, we divided the dataset into 10 k-folds for cross-validation to observe the effects of different k values, and identify the optimal value to prevent overfitting and underfitting of the training set. 

From the graph below, we conclude that k=5 is gives the best estimate for our data set.


```
def knn(full_data, features, L, K):
    
    ... ...
     
    return average_rmse
```

![img2](https://user-images.githubusercontent.com/42630569/52907500-a1b2d880-3217-11e9-90b7-a30c45b300bc.png)



### Forward selection of features

To determine the features to include, linear regression was run and cross-validation was used to calculate the RMSE for each iteration. Combining both tests, lowest RMSE of 3.52 is achieved with features LSTAT, RM, DIS, PTRATIO, NOX, RAD, B used and a k-value of 5.

![img3](https://user-images.githubusercontent.com/42630569/52907534-55b46380-3218-11e9-98e9-0b5c80b891b4.png)

## Part II : Linear Regression

First, RMSE score with varying flexibility in the linear regression model was tested. Using just one variable RM, I compared the RMSE of predictions with RM, and RMSE of predictions with both RM and RM-squared.

![img4](https://user-images.githubusercontent.com/42630569/52907564-f30f9780-3218-11e9-9d75-d7bdffd66173.png)

| LR model | CV RMSE |
| ---         |     ---     | 
| RM   | 6.60 |
| RM + RM-squared   | 6.17  | 


## Gradient descent

Gradient descent was written from scratch to compare the number of iterations required to achieve minimum loss, with different learning rates. The processing times were also recorded.

The code is as follows:
```
def multivariate_ols(xvalue_matrix, yvalues, R=0.01, MaxIterations=10000):
    
    xvalue_matrix = np.array(xvalue_matrix)
    yvalues = np.array(yvalues)
    
    start_time = time.time()
    
    num_xvar = xvalue_matrix.shape[1]
    num_samples = xvalue_matrix.shape[0]
    alpha = 30    # random alpha
    beta_array = np.ones((1, num_xvar))
    x_transpose = xvalue_matrix.T

    error = alpha + (np.dot(beta_array, x_transpose)).T - yvalues
    errorsq = error**2
    sum_errorsq = np.sum(errorsq, axis=0)
    cost = 1/(2*num_samples) * sum_errorsq
    
    count = 0
    diff_a = 1    # dummy
    diff_b = 1    # dummy
    
    while count < MaxIterations and abs(R*diff_a) > 1e-8 and abs(R*diff_b) > 1e-8:  
        # differential of alpha
        diff_a = 1/num_samples * np.sum(error, axis=0)
        alpha = np.array(alpha - R*diff_a)    # update alpha

        # differential of beta array
        for i in range(num_xvar):
            x_list = np.reshape(np.array(xvalue_matrix[:,i]), (xvalue_matrix.shape[0],1))
            diff_b = 1/num_samples * (np.sum(np.array(error * x_list), axis=0))
            beta_array[0][i] = beta_array[0][i] - R*diff_b    # update each beta
    
        error = alpha + (np.dot(beta_array, x_transpose)).T - yvalues
        count += 1
        
    # calculate new cost function
    post_errorsq = error**2
    post_sum_errorsq = np.sum(post_errorsq, axis=0)
    post_cost = 1/(2*num_samples) * post_sum_errorsq
    
    print("No. of Iterations: {}".format(count))
    print("Time taken: {:.2f} seconds".format(time.time() - start_time))
    print("Cost function after iterations {}.".format(post_cost[0]))
    print("alpha : " + str(alpha))
    print("beta : " + str(beta_array))
    return alpha, beta_array
   ```
   
   Summary of the results is tabulated:
   
   | Variables used | Normalized | Learning rate | Time taken | No. of iterations| Cost function | 
   | -----          | -----       |  ----- | -----  | -----  | ----- |
   | RM |  No | 0.01 | 3.02 seconds | 97,358 | 21.8 |
   | RM |  No | 0.02 | 1.56 seconds | 51,559 | 21.8 |
   | RM |  No | 0.03 | 1.31 seconds | 35,495 | 21.8 |
   | RM |  No | 0.04 | 0.83 seconds | 27,218 | 21.8 |
   | CRIM, RM |  Yes | 0.1 | 0.02 seconds | 174 | 19.4 |
   | CRIM, RM |  Yes | 0.01 | 0.16 seconds | 1576 | 19.4 |
   | CRIM, RM |  Yes | 0.001 | 1.04 seconds | 13518 | 19.4 |
   | CRIM, RM |  No | 0.1 | 0.02 seconds | - | Did not converge |
   | CRIM, RM |  No | 0.01 | 9.41 seconds | 103005 | 19.4 |
   

## Increasing number of features

With K number of features, all possible pair-interactions are added to give a total of K+(K*(K+1))/2 number of features. Normalizing the features and taking 33% as testing data, train RMSE and test RMSE of 3.40 and 3.98 are obtained, proving an overfit on the training data. 

## Ridge regularization

To prevent overfitting, ridge regularization is run and we can observe the effect of magnitude of lambda on each coefficient.

![img5](https://user-images.githubusercontent.com/42630569/52910718-7c8e8c00-3250-11e9-8887-ac4db3c4dc83.png)
