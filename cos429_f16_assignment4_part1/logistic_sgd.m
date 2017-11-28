%
% Princeton University, COS 429, Fall 2016
%
% logistic_sgd.m
%   Performs logistic regression via Stochastic Gradient Descent
%
% Inputs:
%   X: datapoints (one per row, should include a column of ones
%                  if the model is to have a constant)
%   z: labels (0/1)
%   num_epochs: number of epochs to train over
% Output:
%   params: vector of parameters 
%

function params = logistic_sgd(X, z, num_epochs)

    [num_pts, num_vars] = size(X);

    % Initial (random) estimate of params.
    mean = 0;
    sigma = 1 / sqrt(num_vars / 2);
    params = normrnd(mean, sigma, num_vars, 1);

    % Loop over epochs
    for ep = 1:num_epochs

        % Permute the data rows
        permutation = randperm(num_pts);
        X = X(permutation, :);
        z = z(permutation);
        
        % Iterate over the points
        for i = 1:num_pts
            gradient = sum((sigmoid(X*params) - z).^2);
            params = params - (1./ep)*gradient;
        end

    end

end

function sig = sigmoid(z)
    sig = 1./exp(z*-1);
end

