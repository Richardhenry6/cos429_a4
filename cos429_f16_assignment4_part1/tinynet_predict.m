%
% Princeton University, COS 429, Fall 2016
%
% tinynet_predict.m
%   Given a tinynet model and some new data, predicts classification
%
% Inputs:
%   X: datapoints (one per row, should include a column of ones
%                  if the model is to have a constant)
%   params: vector of parameters 
% Output:
%   z: predicted labels (0/1)
%
function z = tinynet_predict(X, net)
    hidden_layer_count = net('hidden_layer_count');
    [example_count, ~] = size(X);
    z_hat = zeros(example_count, 1);
    act = containers.Map('KeyType', 'int32', 'ValueType', 'any');
    for ei = 1:example_count
        x = X(ei, :);
        % Set z_hat(ei) by propogating x through the network.
        % This task is a warm-up: a superset of the functionality required
        % here is already implemented in the full_forward_pass() function
        % in tinynet_sgd.m
        W_1 = net('hidden-1-W');
        b_1 = net('hidden-1-b');
        act(1) = [relu(fully_connected(x, W_1, b_1))];
        for i = 2:hidden_layer_count
            W = net(sprintf('hidden-%i-W', i));
            b = net(sprintf('hidden-%i-b', i));
            % Apply the ith hidden layer and relu and update x.
            act(i) = relu(fully_connected(act(i-1), W, b));
        end
    
        W = net('final-W');
        b = net('final-b');
        x = fully_connected(act(hidden_layer_count), W, b);
        z_hat(ei) = logistic(x);
        
    end
    z = (z_hat > 0.5);
end
