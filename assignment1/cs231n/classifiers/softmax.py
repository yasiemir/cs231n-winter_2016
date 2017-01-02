import numpy as np
from random import shuffle

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


  num_train = X.shape[0]
  num_class = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  #numeric stability


  scores = scores - np.max(scores, axis = 1)[:,np.newaxis]
  scores = np.exp(scores)
  all_class_proba = scores / np.sum(scores, axis = 1)[:, np.newaxis]
  true_class_proba = all_class_proba[range(num_train), y]

  all_class_score = all_class_proba
  all_class_score[range(num_train), y] -= 1



  loss = np.sum(-np.log(true_class_proba))
  loss /= num_train
  loss += 0.5 * reg * np.sum(W*W)

  dW = np.dot(X.T, all_class_score)
  dW /= num_train
  dW += reg*W



  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  num_train = X.shape[0]
  num_class = W.shape[1]

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = X.dot(W)
  #numeric stability


  scores = scores - np.max(scores, axis = 1)[:,np.newaxis]
  scores = np.exp(scores)
  all_class_proba = scores / np.sum(scores, axis = 1)[:, np.newaxis]
  true_class_proba = all_class_proba[range(num_train), y]

  all_class_score = all_class_proba
  all_class_score[range(num_train), y] -= 1



  loss = np.sum(-np.log(true_class_proba))
  loss /= num_train
  loss += 0.5 * reg * np.sum(W*W)

  dW = np.dot(X.T, all_class_score)
  dW /= num_train
  dW += reg*W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW
