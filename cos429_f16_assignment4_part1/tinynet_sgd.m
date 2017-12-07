%
% Princeton University, COS 429, Fall 2016
%
% tinynet_sgd.m
%   Trains a tiny (2 hidden-node + 1 output) network with SGD
%
% Inputs:
%   X: datapoints (one per row, should include a column of ones
%                  if the model is to have a constant)
%   z: ground-truth labels (0/1)
%   layers: Kx1 list. K = the number of hidden layers, excluding the
%   final post-relu layer. The value of each layer (first-layer:end=1:end)
%   is the number of hidden layer neurons.
%   epoch_count: number of epochs to train over
% Output:
%   params: vector of parameters 
function net = tinynet_sgd(X, z, layers, epoch_count)

    [example_count, feature_count] = size(X);
  
    % Randomly initialize the parameters.
    net = initialize_net(layers, feature_count);
    hidden_layer_count = net('hidden_layer_count');
    % The value at key i is the intermediate output of the network at
    % layer i. The type is 'any' because the neuron count is variable.
    activations = containers.Map('KeyType', 'int32', 'ValueType', 'any');
    % For each epoch, train on all data examples.
    for ep = 1:epoch_count
        fprintf('Starting epoch %i of %i...\n', ep, epoch_count);
        learning_rate = 0.1/ep;
        % Randomly shuffle the examples so they aren't applied in the
        % same order every epoch.
        permutation = randperm(example_count);
        X = X(permutation, :);
        z = z(permutation);

        % Train on each example by making a prediction and doing backprop
        % Note that in a full-featured software package like TensorFLow, 
        % you would pull a batch of images, and all the functions you
        % implemented here would be efficiently batched to reduce repeated
        % work. If your brain is melting thinking through some of the
        % gradients, remember that at least x is a vector, not a matrix. We
        % do care.
        for i = 1:example_count

            % For simplicity set the '0-th' layer activation to be the
            % input to the network. activations(1...hidden_layer_count) are
            % the outputs of the hidden layers of the network before relu.
            % activations(hidden_layer_count+1) is the output of the output
            % layer. The logistic layers and layer don't need to be stored
            % here to compute the derivatives and updates at the end of
            % the network.
            % You will at some point want to access the post relu
            % activations. Just call relu on the activations to get that.
            % Faster networks would cache them, but memory management here
            % is already bad enough that it won't matter much.
            activations(0) = X(i, :);
            
            % Forward propagation pass to evaluate the network
            % z_hat is the output of the network; activations has been
            % updated in-place to contain the network outputs of layer i
            % at activations(i).
            z_hat = full_forward_pass(X(i, :), net, activations);

            % Backwards pass to evaluate gradients at each layer and update
            % weights. First compute dL/dz_hat here, then use the backprop
            % functions to get the gradients for earlier weights as you
            % move backwards through the network, updating the weights
            % at each step.
            % Note: Study full_forward_pass, defined below, for an example
            % of how the information you need to access can be referenced.
            % [net] contains the current network weights (and is a handle
            % object, so please be careful when editing it). 
            % [activations] contains the responses of the network at each
            % layer; activations(0) is the input,
            % activations(1:hidden_layer_count) is the output of each layer
            % before relu has been applied.
            % activations(hidden_layer_count+1) is the final output before
            % it is squished with the logistic function.
            dLdz_hat = 2*(z_hat - z(i));
            %out = relu(X(i,:));
            %out(out<=0)= 0;
            %out(out>0)= 1;
            %dLdzz = out.*X(i,:)
            dxval = logistic_backprop(dLdz_hat,z_hat);
           % net(sprintf('hidden-%i-W', example_count-1)) = dxval;
           % dLdX2=relu_backprop(dxval,X(i,:));
           % [dx dw db] = fully_connected_backprop(dLdX2,activations(i),net('final-W'));        
        end
    end
end

% Full forward pass, caching intermediate 
function z_hat = full_forward_pass(example, net, activations)
    hidden_layer_count = net('hidden_layer_count');
    x = example;
    
    W_1 = net('hidden-1-W');
    b_1 = net('hidden-1-b');
    activations(1) = relu(fully_connected(x, W_1, b_1));
    for i = 2:hidden_layer_count
       W = net(sprintf('hidden-%i-W', i));
       b = net(sprintf('hidden-%i-b', i));
       % Apply the ith hidden layer and relu and update x.
       activations(i) = relu(fully_connected(activations(i-1), W, b));
    end
    
    W = net('final-W');
    b = net('final-b');
    % Apply the final layer, and then the sigmoid to get zhat.
    % Ignore the unused warning for activations - it is a handle object, so
    % it is pass by reference.
    x = fully_connected(activations(hidden_layer_count), W, b);
    activations(hidden_layer_count+1) = x;
    z_hat = logistic(x);
end

function net = initialize_net(layers, feature_count)
% Initializes the network with matrices of the correct size containing
% random weights scaled as in He et al. Also initializes the biases to
% 0-vectors. Reading this function may be helpful for understanding how
% the network is structured, but you don't need to edit it. 
    [hidden_layer_count, ~] = size(layers);
    net = containers.Map('KeyType', 'char', 'ValueType', 'any');
    net('layers') = layers;
    net('hidden_layer_count') = hidden_layer_count;
    
    % Initialize hidden layers
    for i = 1:hidden_layer_count
        if i == 1
            layer_rows = feature_count;
        else
            layer_rows = layers(i-1);
        end
        layer_cols = layers(i);
        layer_sigma = sqrt(2.0/layer_rows);
        
        W = zeros(layer_rows, layer_cols);
        for col = 1:layer_cols
            W(:, col) = layer_sigma * normrnd(0, 1, layer_rows, 1);
        end
        net(sprintf('hidden-%i-W', i)) = W;
        net(sprintf('hidden-%i-b', i)) = zeros(1, layer_cols);
    end
    % Initialize final layer
    layer_size = layers(end);
    layer_sigma = 1.0 / sqrt(layer_size);
    net('final-W') = layer_sigma * normrnd(0, 1, layer_size, 1);
    net('final-b') = zeros(1, 1);
end

