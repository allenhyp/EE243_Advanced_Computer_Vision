function new_theta = apply_gradients(x,l,theta,lr)

% x is a matrix of size n_samples x n_feature
% l is a vector of size n_samples x n_class
% theta is a matrix of size n_feature x n_class
% lr is the learning rate

% FILL IN
h = x*theta;
a = activation_function(h);
loss = a - l;
gradient = x.'*loss/length(l);
new_theta = theta - lr*gradient;
