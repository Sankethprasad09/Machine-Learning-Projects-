import numpy as np
np.random.seed(42)
import matplotlib.pyplot as plt
import logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

# GLOBAL PARAMETERS FOR STOCHASTIC GRADIENT DESCENT
step_sizes = [1.0, 0.1, 0.01, 0.00001]

def main():

  # Load the training data
  logging.info("Loading data")
  X_train, y_train, X_test = loadData()

  logging.info("\n---------------------------------------------------------------------------\n")
  
  #Augment training and testing data with bias term
  X_train_bias = dummyAugment(X_train)
  X_test_bias = dummyAugment(X_test)
  
  # Fit a logistic regression model on train and plot its losses
  logging.info("Training logistic regression model (No Bias Term)")
  w, losses = trainLogistic(X_train,y_train)
  y_pred_train = X_train@w >= 0
  
  logging.info("Learned weight vector: {}".format([np.round(a,4)[0] for a in w]))
  logging.info("Train accuracy: {:.4}%".format(np.mean(y_pred_train == y_train)*100))
  
  logging.info("\n---------------------------------------------------------------------------\n")

  # Fit a logistic regression model on train and plot its losses
  logging.info("Training logistic regression model (Added Bias Term)")
  w, bias_losses = trainLogistic(X_train_bias, y_train)
  y_pred_train = X_train_bias @ w >= 0
  
  logging.info("Learned weight vector: {}".format([np.round(a,4)[0] for a in w]))
  logging.info("Train accuracy: {:.4}%".format(np.mean(y_pred_train == y_train)*100))
  
  logging.info("\n---------------------------------------------------------------------------\n")

  

  # Loop over step sizes and train the model each time
  for step_size in step_sizes:
        logging.info(f"Training logistic regression model with step size: {step_size}")
        w, losses = trainLogistic(X_train_bias, y_train, step_size=step_size)
        y_pred_train = X_train_bias @ w >= 0
        train_accuracy = np.mean(y_pred_train == y_train) * 100
        logging.info(f"Train accuracy with step size {step_size}: {train_accuracy:.4}%")
        
        # Plot losses after each training to avoid overwriting the plot
        plt.plot(np.arange(len(losses)), losses, label=f"Step Size {step_size}")
        

  # Finalize and show the plot after the loop
  plt.figure(figsize=(16,9))
  plt.plot(range(len(losses)), losses, label="No Bias Term Added")
  plt.plot(range(len(bias_losses)), bias_losses, label="Bias Term Added")
  plt.title("Logistic Regression Training Curve")
  plt.xlabel("Epoch")
  plt.ylabel("Negative Log Likelihood")
  plt.legend()
  plt.show()

  logging.info("\n---------------------------------------------------------------------------\n")

  logging.info("Running cross-fold validation for bias case:")

  # Perform k-fold cross
  for k in [2,3,4, 5, 10, 20, 50]:
    cv_acc, cv_std = kFoldCrossVal(X_train_bias, y_train, k)
    logging.info("{}-fold Cross Val Accuracy -- Mean (stdev): {:.4}% ({:.4}%)".format(k,cv_acc*100, cv_std*100))

  # Load the test data with the same preprocessing as training data
  _, _, X_test = loadData()

  # Fit a logistic regression model on the train data
  w, _ = trainLogistic(X_train_bias, y_train)

  # Make predictions on the test data using the bias term
  y_pred_test = X_test_bias @ w >= 0.5

  # Format the predictions for submission
  with open("submission.csv", "w") as f:
    f.write("id,type\n")
    for i, prediction in enumerate(y_pred_test):
        # Ensure that the prediction is treated as a scalar by using [0] indexing.
        f.write(f"{i},{int(prediction[0])}\n")

  # Print a message indicating that the predictions are saved
  print("Test predictions saved to submission.csv")


######################################################################
# Q3.1 logistic 
######################################################################
# Given an input vector z, return a vector of the outputs of a logistic
# function applied to each input value
#
# Input: 
#   z --   a n-by-1 vector
#
# Output:
#   logit_z --  a n-by-1 vector where logit_z[i] is the result of 
#               applying the logistic function to z[i]
######################################################################
def logistic(z):
    """
    Given an n x 1 input vector z, return a n x 1 vector such that the i'th element of the output is σ(zi).
    
    Args:
    z (numpy.ndarray): n x 1 input vector
    
    Returns:
    logit_z (numpy.ndarray): n x 1 vector where logit_z[i] is σ(zi)
    """
    sigmoid = 1 / (1 + np.exp(-z))
    return sigmoid


######################################################################
# Q3.2 calculateNegativeLogLikelihood 
######################################################################
# Given an input data matrix X, label vector y, and weight vector w
# compute the negative log likelihood of a logistic regression model
# using w on the data defined by X and y
#
# Input: 
#   X --   a n-by-d matrix of examples where each row
#                   corresponds to a single d-dimensional example
#
#   y --    a n-by-1 vector representing the labels of the examples in X
#
#   w --    a d-by-1 weight bector
#
# Output:
#   nll --  the value of the negative log-likelihood
######################################################################
def calculateNegativeLogLikelihood(X, y, w):
    """
    Given an n x d input data matrix X where each row represents one datapoint, a n x 1 label vector y,
    and d x 1 weight vector w, compute the negative log likelihood of a logistic regression model
    that applies w on the data defined by X and y.
    
    Args:
    X (numpy.ndarray): n x d input data matrix
    y (numpy.ndarray): n x 1 label vector
    w (numpy.ndarray): d x 1 weight vector
    
    Returns:
    nll (float): The value of the negative log-likelihood
    """
    # Calculate the linear combination of weights and features
    z = np.dot(X, w)
    
    # Calculate the logistic function for each data point
    sigmoid_z = logistic(z)
    
    # Add a small constant inside the logarithm to prevent overflow when probability is close to zero
    epsilon = 1e-7
    
    # Compute the negative log-likelihood as per Eq. 8
    nll = -np.sum(y * np.log(sigmoid_z + epsilon) + (1 - y) * np.log(1 - sigmoid_z + epsilon))
    
    return nll

######################################################################
# Q4 trainLogistic
######################################################################
# Given an input data matrix X, label vector y, maximum number of 
# iterations max_iters, and step size step_size -- run max_iters of 
# gradient descent with a step size of step_size to optimize a weight
# vector that minimizies negative log-likelihood on the data defined
# by X and y
#
# Input: 
#   X --   a n-by-d matrix of examples where each row
#                   corresponds to a single d-dimensional example
#
#   y --    a n-by-1 vector representing the labels of the examples in X
#
#   max_iters --   the maximum number of gradient descent iterations
#
#   step_size -- the step size (or learning rate) for gradient descent
#
# Output:
#   w --  the d-by-1 weight vector at the end of training
#
#   losses -- a list of negative log-likelihood values for each iteration
######################################################################
def trainLogistic(X, y, max_iters= 2000, step_size=0.0001):
    # Initialize our weights with zeros
    w = np.zeros((X.shape[1], 1))
    # Keep track of losses for plotting
    losses = [calculateNegativeLogLikelihood(X, y, w)]

    for i in range(max_iters):
        # Compute the logistic function applied to Xw
        sigmoid_Xw = logistic(np.dot(X, w))

        # Compute the gradient using Eq. 9
        w_grad = np.dot(X.T, sigmoid_Xw - y)

        # Take the update step in gradient descent
        w = w - step_size * w_grad

        # Calculate the negative log-likelihood with the updated w
        losses.append(calculateNegativeLogLikelihood(X, y, w))

    return w, losses



######################################################################
# Q5 dummyAugment
######################################################################
# Given an input data matrix X, add a column of ones to the left-hand
# side
#
# Input: 
#   X --   a n-by-d matrix of examples where each row
#                   corresponds to a single d-dimensional example
#
# Output:
#   aug_X --  a n-by-(d+1) matrix of examples where each row
#                   corresponds to a single d-dimensional example
#                   where the the first column is all ones
#
######################################################################
def dummyAugment(X):
    """
    Given an input data matrix X, add a column of ones to the left-hand side.
    
    Args:
    X (numpy.ndarray): n x d matrix of examples
    
    Returns:
    aug_X (numpy.ndarray): n x (d+1) matrix with an added column of ones
    """
    ones_column = np.ones((X.shape[0], 1))
    aug_X = np.hstack((ones_column, X))
    return aug_X





##################################################################
# Instructor Provided Code, Don't need to modify but should read
##################################################################

# Given a matrix X (n x d) and y (n x 1), perform k fold cross val.
def kFoldCrossVal(X, y, k):
  fold_size = int(np.ceil(len(X)/k))
  
  rand_inds = np.random.permutation(len(X))
  X = X[rand_inds]
  y = y[rand_inds]

  acc = []
  inds = np.arange(len(X))
  for j in range(k):
    
    start = min(len(X),fold_size*j)
    end = min(len(X),fold_size*(j+1))
    test_idx = np.arange(start, end)
    train_idx = np.concatenate( [np.arange(0,start), np.arange(end, len(X))] )
    if len(test_idx) < 2:
      break

    X_fold_test = X[test_idx]
    y_fold_test = y[test_idx]
    
    X_fold_train = X[train_idx]
    y_fold_train = y[train_idx]

    w, losses = trainLogistic(X_fold_train, y_fold_train)

    acc.append(np.mean((X_fold_test@w >= 0) == y_fold_test))

  return np.mean(acc), np.std(acc)


# Loads the train and test splits, passes back x/y for train and just x for test
def loadData():
  train = np.loadtxt("/Users/sankethkaruturi/Desktop/CS 434 HW2/train_cancer.csv", delimiter=",")
  test = np.loadtxt("/Users/sankethkaruturi/Desktop/CS 434 HW2/test_cancer_pub.csv", delimiter=",")
  
  X_train = train[:, 0:-1]
  y_train = train[:, -1]
  X_test = test
  
  return X_train, y_train[:, np.newaxis], X_test   # The np.newaxis trick changes it from a (n,) matrix to a (n,1) matrix.


main()
