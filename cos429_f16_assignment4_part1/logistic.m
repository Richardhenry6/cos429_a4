function y = logistic(x)
%
% The logistic "sigmoid" function
%
% x and y are both 1x1 doubles.
    y = 1/(1+exp(-x));
end