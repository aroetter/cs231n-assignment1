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

    # now compute the gradient of the loss wrt the i,jth weight (the
    # weight for the ith data pt on the jth class)
    # i just manually took the derivative of:
    # L_i = -f(yi) + log(sum over j(e^f(j))
    # which yields
    # dW_ij = 1{j=y(i)} * -1 * x_i                  # dW_term1
    #         + 1/(sum over j e^f(j)) * e^f(j)*x_i  # dW_term2
    # 
    for j in xrange(num_classes):
      dW[:, j] += (1 / denom) * math.exp(scores[i, j]) * X[i]
      if j == y[i]:
        dW[:, j] -= X[i]
        
  loss /= num_train
  dW /= num_train

  # add in the regularization contributions
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W

  return loss, dW

def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  num_train = X.shape[0]
  num_classes = W.shape[1]
  # gives an N x C matrix.
  scores = X.dot(W)
  # make largest value here zero, just for numeric stability
  scores -=np.max(scores)
  expscores = np.exp(scores)

  denoms = expscores.sum(axis=1)
  numers = expscores[range(num_train), y]
  loss = -1 * np.log(numers / denoms).sum()
  ########

  #Now compute dW
  # remember from above, the derivative with respect to w_ij, which we call
  # dW_ij, is 2 terms. the first only triggers if the ith data pt is in the jth
  # class, the 2nd term is a product of 3 things (see comments above in the naive
  # impl to see what i mean by dW_term1 and dW_term2
  correct_class = np.zeros((num_train, num_classes))
  correct_class[xrange(num_train),y] = -1
  dW_term1 = X.T.dot(correct_class)
  dW_term2 = (X.T / denoms).dot(expscores)
  dW = dW_term1 + dW_term2

  loss /= num_train
  dW /= float(num_train)

  # add in the regularization contributions
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  
  return loss, dW

