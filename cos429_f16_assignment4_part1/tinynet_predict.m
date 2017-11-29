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
    for ei = 1:example_count
        x = X(ei, :);
        % Set z_hat(ei) by propogating x through the network.
        % This task is a warm-up: a superset of the functionality required
        % here is already implemented in the full_forward_pass() function
        % in tinynet_sgd.m
        % TODO: Implement me!
        assert(false, 'Unimplemented: tinynet_predict!');
    end
    z = (z_hat > 0.5);
end
