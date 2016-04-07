import numpy as np
from random import shuffle
import math

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
  # gives an N x C matrix.
  scores = X.dot(W)
  # make largest value here zero, just for numeric stability
  scores -= np.max(scores)
  
  num_train = X.shape[0]
  num_classes = W.shape[1]
  num_datapt_dims = X.shape[1]
  for i in xrange(num_train):
    # compute the loss for the ith datapt
    denom = 0.0
    for j in xrange(num_classes):
      denom += math.exp(scores[i, j])
    l_i = -1 * scores[i, y[i]] + math.log(denom)
    loss += l_i

    # try to compute dW_ij.
    # i just manually took the derivative of:
    # L_i = -f(yi) + log(sum over j(e^f(j))
    # i think that yields
    # dW (wrt w_dj) = x_d (if j is the correct class (y_i)
    #                 + 1/(sum over j e^f(j)) * e^f(j)*d_d
    # here x_d is the dth dimension of the ith data pt x (really called x_i_d)
    for j in xrange(num_classes):
      dW[:, j] += (1 / denom) * math.exp(scores[i, j]) * X[i]
      if j == y[i]:
        dW[:, j] -= X[i]
        
  loss /= num_train
  dW /= num_train

  # add in the regularization loss
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  pass

  # okay, now compute the dW with loops.
  # W is a D x C matrix, so is dW.
  
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

  
  ########
  num_train = X.shape[0]
  # gives an N x C matrix.
  scores = X.dot(W)
  # for numerical stability
  expscores = np.exp(scores)
  denoms = expscores.sum(axis=1)
  numers = expscores[range(num_train), y]
  vector_loss = -1 * np.log(numers / denoms).sum()
  vector_loss /= num_train
  print "vector loss (ignoring reg): ", vector_loss
  ########
  
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################



  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

