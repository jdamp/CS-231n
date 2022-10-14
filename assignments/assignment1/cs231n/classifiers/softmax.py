import numpy as np


def stable_softmax(x, axis=-1):
    """
    Inputs:
    - x: A numpy array
    - axis: (int) Over which axis the summation should be done
    """
    z = x - np.max(x)
    exp = np.exp(z)
    return exp/np.sum(exp, axis, keepdims=True)


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
    n, _ = X.shape
    _, c = W.shape
    for i in range(n):
        softmax = stable_softmax(X[i, :].dot(W))
        loss -= np.log(softmax[y[i]])
        softmax[y[i]] -= 1
        dW += np.outer(X[i], softmax)

    # Add regularization
    loss = loss / n + reg * np.sum(W**2)
    dW = dW / n + 2 * reg * W
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
    n, _ = X.shape

    softmax = stable_softmax(X.dot(W))  # shape (N, C)
    # pick one entry according to the label in y
    softmax_y = softmax[range(n), y]   # shape (N,)
    loss = -np.sum(np.log(softmax_y)) / n + reg * np.sum(W**2)

    # change softmax for gradient
    softmax[range(n), y] -= 1
    dW = X.T.dot(softmax) / n + 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
