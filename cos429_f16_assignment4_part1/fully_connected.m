% The forward layer for a hidden neuron. 
function U = fully_connected(x, W, b)
% x: a row vector with shape [1, feature_count].
% W: a matrix with shape [feature_count, neuron_count] containing the
% parameters of the hidden layer.
% U: a row vector with shape [1, neuron_count] containing the network
% responses at the hidden layer.
    % TODO: Implement me!
    U = x * W + b;
end
