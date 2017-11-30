function y = relu(x)
% x: a 2-D double array with arbitrary shape.
% [y]: relu(x) as described in class applied elementwise.
% TODO: Implement me!
    x(x<0) = 0;
    y = x;
end
