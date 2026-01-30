from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    num_classes = W.shape[1]

    for i in range(num_train):
      scores = X[i].dot(W)   # calculate the scores for the i-th example for all classes

      # shift scores so that the max value is 0. This prevents overflow when exponentiating, without changing the final result.
      exp_scores = np.exp(scores - np.max(scores))   
      
      # sum the exponentiated scores to get the denominator of the softmax function 
      sum_exp_scores = np.sum(exp_scores)

      softmax = exp_scores / sum_exp_scores    # calculate the probabilities for all other classes  
      loss -= np.log(softmax[y[i]])            # cross entropy of the correct class

      # the gradient with respect to the scores is the probability vector,
      # minus 1 at the position of the correct class.
      softmax[y[i]] -= 1  # update for gradient

      # backpropagate this gradient to the weights W.
      dW += np.outer(X[i], softmax)   # numpy outer function to calculate outer product directly

    loss /= num_train                 # average the data loss over all training examples
    loss += reg * np.sum(W * W)       # add the L2 regularization loss

    dW /= num_train                   # average the gradient over all training examples
    dW += 2 * reg * W                 # add the gradient of the L2 regularization

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    scores = X.dot(W)  # compute all scores
    
    # compute exponentials and ensure numerical stability by shifting scores by substracting max in each row
    exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True)) 

    softmax = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)    # compute row-wise softmax probabilities
    correct_class_probs = softmax[np.arange(num_train), y]              # compute loss: -log(probability of correct class)
    
    loss = -np.sum(np.log(correct_class_probs))  # sum cross entropies as loss

    loss /= num_train               # average the loss
    loss += reg * np.sum(W * W)     # add regularization
    
    # update for gradient
    softmax[np.arange(num_train), y] -= 1  # subtract 1 from correct class probabilities
    
    # backpropagate to weights
    dW = X.T.dot(softmax)  

    dW /= num_train     # average the gradient
    dW += 2 * reg * W   # add regularization gradient

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
