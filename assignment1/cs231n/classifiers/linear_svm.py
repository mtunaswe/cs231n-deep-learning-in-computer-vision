from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                #########################################################
                #                    START OF CHANGE                    #
                #########################################################
                # Gradient for incorrect class j (contributes positively to loss)
                dW[:, j] += X[i]   
                # Gradient for correct class y[i] (contributes negatively)
                dW[:, y[i]] -= X[i] 
                #########################################################
                #                     END OF CHANGE                     #
                #########################################################

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    dW /= num_train 

    # Add regularization gradient.
    dW += 2 * reg * W 

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]  # number of samples
    scores = X.dot(W)       # compute all scores: shape (N, C)
    
    # get the scores of correct classes for each example: shape (N, 1)
    correct_class_scores = scores[np.arange(num_train), y]
    
    # compute margins: shape (N, C)
    # broadcasting: correct_class_scores[:, np.newaxis] has shape (N, 1)
    margins = np.maximum(0, scores - correct_class_scores[:, np.newaxis] + 1)
    
    # remove out the correct class in the loss (set its margin to 0)
    margins[np.arange(num_train), y] = 0
    
    loss = np.sum(margins) / num_train  # sum all margins and average
    loss += reg * np.sum(W * W)         # add regularization

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    binary = margins 
    binary[margins > 0] = 1  # create a binary mask for margins > 0
   
    row_sum = np.sum(binary, axis=1)               # count how many classes contributed to the loss for each example
    binary[np.arange(num_train), y] = -row_sum     # this tells us how many times to subtract X[i] from the correct class gradient
    
    dW = X.T.dot(binary) # compute gradient: dW = X^T * binary

    dW /= num_train       # average over training examples
    dW += 2 * reg * W     # add regularization gradient

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
