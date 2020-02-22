% Generate a 10 dimension matrix of uniformly distributed random numbers between 0 and 1.
% number of training examples
N = 100;
% dimensionality of each example (before adding the 1)
d = 10;
% the number of times to repeat the experiment
num_samples = 1000;

[num_iters, bounds_minus_ni] = perceptron_experiment(N, d, num_samples);







