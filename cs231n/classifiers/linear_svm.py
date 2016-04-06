import numpy as np
from random import shuffle

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]

    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        # we'll end up adding X[i] to the jth column for every X[i] that
        # contributed to the loss
        dW[:, j] += X[i]
        # this will trigger K times if K classes were wrong enough to add loss
        dW[:, y[i]] -= X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  # that contribues to the gradient as well
  dW += reg * W

  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # an array of N x C, where entry is the loss from the nth data pt due
  # to it's score against the cth class
  scores = X.dot(W) # yields 500,10
  #print "Sanity check scores=", scores.shape

  correct_label_scores = scores[range(scores.shape[0]), y]

  # now reshape that to be wide, subtract from all items
  # print "scores shape is: ", scores.shape # should be 500, 10
  # print "Correct_label_scores shape is: ", correct_label_scores.shape # should be 500
  # subtract correct scores (as a column vector) from every cell
  # TODO patch up w/o tmp variable
  tmp = np.reshape(correct_label_scores, (-1, 1))
  print "tmp shape is (should be 500,1)): ", tmp.shape
  scores_diff = scores - tmp
  
  # add 1 for the margin.
  scores_diff += 1
  
  # now zero out all the loss scores for the correct classes.
  ##print "correct classes for top 2 rows are:"
  ##print y[:2]
  ##print "top 2 rows of scores_diff is..."
  ##print scores_diff[:2,:]
  scores_diff[range(scores_diff.shape[0]), y] = 0

  ##print "top 2 rows of scores_diff is..."
  print scores_diff[11:15,:]

  # now zero out all elements less than zero.
  indexes_of_neg_nums = np.nonzero(scores_diff < 0)
  scores_diff[indexes_of_neg_nums] = 0
  
  #now sum over all dimensions
  loss = scores_diff.sum()
  num_train = X.shape[0]
  loss /= num_train
  
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
