from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # first, we create a list of all layer dimensions, 
        # from input to output, including hidden dims.
        layer_dimensions = [input_dim] + hidden_dims + [num_classes]

        # loop over the first layer (1) up to and including the last layer
        for l in range(1, self.num_layers + 1):
            # get the input and output size for this layer 'l'
            # l=1: (layer_dimensions[0], layer_dimensions[1]) -> (input_dim, 100)
            # l=2: (layer_dimensions[1], layer_dimensions[2]) -> (100, 50)
            # l=3: (layer_dimensions[2], layer_dimensions[3]) -> (50, num_classes)
            input_size = layer_dimensions[l - 1]
            output_size = layer_dimensions[l]
            
            # initialize weights with values from a standard normal distribution
            # then multiply by weight_scale to control Controls the initial magnitude 
            # of weights. 
            # too large → exploding gradients; 
            # too small → vanishing gradients.
            self.params[f'W{l}'] = np.random.randn(input_size, output_size) * weight_scale
            
            # initialize biases to zero
            self.params[f'b{l}'] = np.zeros(output_size)
            
            # initialize Normalization parameters (gamma and beta)
            # we do this for all layers except the output (last) layer
            if self.normalization is not None and l < self.num_layers:
                self.params[f'gamma{l}'] = np.ones(output_size)   # gamma (scale) is initialized to ones
                self.params[f'beta{l}'] = np.zeros(output_size)   # beta (shift) is initialized to zeros

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.
        
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # FORWARD PASS
        layer_caches = []
        current_input = X

        # pass through L-1 hidden layers (with ReLU)
        for layer_idx in range(1 ,self.num_layers):
            W, b = self.params[f'W{layer_idx}'], self.params[f'b{layer_idx}']

            affine_cache, bn_cache, ln_cache, relu_cache, dropout_cache = None, None, None, None, None
            
            # affine step
            current_out, affine_cache = affine_forward(current_input, W, b)

            # normalization step (Batchnorm or Layernorm)
            if self.normalization is not None:
                gamma = self.params[f'gamma{layer_idx}']
                beta = self.params[f'beta{layer_idx}']
                bn_param = self.bn_params[layer_idx - 1] # This is {"mode":...} or {}
                
                # check for 'mode' key to distinguish batchnorm from layernorm
                if 'mode' in bn_param:
                    current_out, bn_cache = batchnorm_forward(current_out, gamma, beta, bn_param)
                else:
                    # 'mode' not in bn_param, so it must be layernorm
                    current_out, ln_cache = layernorm_forward(current_out, gamma, beta, bn_param)
            
            # ReLU activation step
            current_out, relu_cache = relu_forward(current_out)

            # dropout step
            if self.use_dropout:
                current_out, dropout_cache = dropout_forward(current_out, self.dropout_param)

            # store all caches for the backward pass
            layer_caches.append((affine_cache, bn_cache, ln_cache, relu_cache, dropout_cache))
            
            # the output of this block is the input to the next
            current_input = current_out
        
        # final layer (affine only, no ReLU)
        W_last, b_last = self.params[f'W{self.num_layers}'], self.params[f'b{self.num_layers}']
        scores, out_cache = affine_forward(current_input, W_last, b_last)
        layer_caches.append((out_cache, None, None, None, None)) 

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early.
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the   #
        # scale and shift parameters.                                              #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # BACKWARD PASS
        loss, grad_scores = softmax_loss(scores, y)

        # regularization loss
        for i in range(1, self.num_layers + 1):
            loss += 0.5 * self.reg * np.sum(self.params[f'W{i}'] ** 2)

        # backprop output layer

        # get the cache for the final layer
        (final_affine_cache, _, _, _, _) = layer_caches[self.num_layers - 1]
        
        # perform the affine backward pass
        upstream_grad, dW, db = affine_backward(grad_scores, final_affine_cache)
        
        # store gradients
        grads[f'W{self.num_layers}'] = dW + self.reg * self.params[f'W{self.num_layers}']
        grads[f'b{self.num_layers}'] = db
        
        # backprop hidden layers in reverse order 
        for l in range(self.num_layers - 1, 0, -1):
            # get the cache for this layer (0-indexed list)
            (affine_cache, bn_cache, ln_cache, relu_cache, dropout_cache) = layer_caches[l - 1]

            # dropout backward (if used)
            # check if the dropout cache was populated in the forward pass
            if dropout_cache is not None:
                upstream_grad = dropout_backward(upstream_grad, dropout_cache)

            # ReLU backward
            if relu_cache is not None:
                upstream_grad = relu_backward(upstream_grad, relu_cache)
          
            # normalization backward (if used)
            # check which cache was populated during the forward pass
            if bn_cache is not None:
                upstream_grad, dgamma, dbeta = batchnorm_backward_alt(upstream_grad, bn_cache)
                grads[f'gamma{l}'] = dgamma
                grads[f'beta{l}'] = dbeta
            elif ln_cache is not None:
                upstream_grad, dgamma, dbeta = layernorm_backward(upstream_grad, ln_cache)
                grads[f'gamma{l}'] = dgamma
                grads[f'beta{l}'] = dbeta

            # affine backward
            upstream_grad, dW, db = affine_backward(upstream_grad, affine_cache)
            
            # store gradients
            grads[f'W{l}'] = dW + self.reg * self.params[f'W{l}']
            grads[f'b{l}'] = db
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
