import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:

  conv - relu - 2x2 max pool - affine - relu - affine - softmax

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.

    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype

    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    self.C, self.H, self.W = input_dim
    self.F = num_filters
    self.k = filter_size
    self.hidden_dims = hidden_dim
    self.num_classes = num_classes

    self.params['W1'] = np.random.normal(loc=0, scale=weight_scale, size=(self.F,self.C,self.k,self.k))
    self.params['b1'] = np.zeros((self.F))
    self.params['W2'] = np.random.normal(loc=0, scale=weight_scale, size=(self.F*self.H*self.H/4,self.hidden_dims))
    self.params['b2'] = np.zeros((self.hidden_dims))
    self.params['W3'] = np.random.normal(loc=0, scale=weight_scale, size=(self.hidden_dims, self.num_classes))
    self.params['b3'] = np.zeros((self.num_classes))
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    cache = []
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    h,temp_cache = conv_relu_pool_forward(X, self.params['W1'], self.params['b1'], conv_param, pool_param)
    cache.append(temp_cache)
    # Flatteing is taken care of at affine forward/backrward steps.
    h,temp_cache = affine_relu_forward(h, self.params['W2'], self.params['b2'])
    cache.append(temp_cache)
    scores,temp_cache = affine_forward(h, self.params['W3'], self.params['b3'])
    cache.append(temp_cache)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
      return scores

    loss, grads = 0, {}
    reg_loss = 0
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dJ = softmax_loss(scores, y)
    dJ, grads['W3'], grads['b3'] = affine_backward(dJ, cache.pop())
    dJ, grads['W2'], grads['b2'] = affine_relu_backward(dJ, cache.pop())
    _,  grads['W1'], grads['b1'] = conv_relu_pool_backward(dJ, cache.pop())



    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    for param in self.params:
        if param.startswith('W'):
            reg_loss += np.sum(self.params[param]*self.params[param])
            grads[param] += self.reg * self.params[param]

    loss += 0.5 * self.reg * reg_loss

    return loss, grads


pass
