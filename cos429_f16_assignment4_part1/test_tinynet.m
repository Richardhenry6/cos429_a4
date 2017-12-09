%
% Princeton University, COS 429, Fall 2016
%
% test_tinynet.m
%   Test tinynet on some simple data in 'training_pacman.txt'
%   and 'test_data.txt'
%
function test_tinynet

    % Load the training data
    training = load('training_pacman.txt');
    
    layers = [3;2]; % additional layers should be semicolon separated.
    for i = 1:16
    % Do the training
    example_count = size(training, 1);
    X = training(:, 1:2);
    z = training(:,3);
    epoch_count = 5;
    net = tinynet_sgd(X, z, layers, epoch_count);
    
    % Apply the learned model to the training data and print out statistics
    % (Notice that the training_accuracy = ... line doesn't have a
    % semicolon, hence the value is printed out.)
    predicted = tinynet_predict(X, net);
    training_accuracy = sum(predicted == z) / example_count
    
    % Plot "ground truth" and predictions
    %set(figure(1), 'Name', 'Training ground truth');
    %plot_classes(training, z);
    %set(figure(2), 'Name', 'Training predicted');
    %plot_classes(training, predicted);

    % Apply the learned model to the test data
    testing = load('test_pacman.txt');
    example_count = size(testing, 1);
    X = testing(:, 1:2);%[ones(example_count,1) testing(:,1:2)];
    z = testing(:,3);
    predicted = tinynet_predict(X, net);
    testing_accuracy = sum(predicted == z) / example_count

    % Plot "ground truth" and predictions
    %set(figure(3), 'Name', 'Testing ground truth');
    %plot_classes(testing, z);
    %set(figure(4), 'Name', 'Testing predicted');
    %plot_classes(testing, predicted);
    end
end

% Create a scatterplot of the given data.  It is assumed that the input data
% is 2-dimensional.
%
% Inputs:
%   data: datapoints (one per row, only first two columns used)
%   z: labels (0/1)
%
function plot_classes(data, z)
    positive = data(z > 0, 1:2);
    negative = data(z == 0, 1:2);
    scatter(positive(:,1), positive(:,2), 'red');
    hold on;  % Next scatter command will be added to the same figure
    scatter(negative(:,1), negative(:,2), 'blue');
    hold off;
end

