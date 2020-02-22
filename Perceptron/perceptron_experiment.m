function [num_iters, bounds_minus_ni] = perceptron_experiment(N, d, num_samples)
% perceptron_experiment: Code for running the perceptron experiment in HW1
% Inputs:  N is the number of training examples
%          d is the dimensionality of each example (before adding the 1)
%          num_samples is the number of times to repeat the experiment
% Outputs: num_iters is the # of iterations PLA takes for each sample
%          bound_minus_ni is the difference between the theoretical bound
%                         and the actual number of iterations
%          (both the outputs should be num_samples long)
bounds_minus_ni = zeros(1,num_samples);
for I = 1:num_samples
    w_10 = rand(d,1);
    thres = 0;
    % 11-dimensional weight vector
    w_star = [thres; w_10];

    %generate random training set 
    %each column is a data point vector(x)
    x = ones(d+1,N);
    for i = 1: N
        x(2:11,i) = -1 + (1+1) * rand(d, 1);
    end

    %generate ground truth
    y = ones(1,N);
    for i = 1: N
        y(1,i) = sign(transpose(w_star) * x(:,i));
    end

    data_in = [x;y];
    
    [w, num_iters(I)] = perceptron_learn(transpose(data_in));
    R = -Inf;
    for i = 1:N
        for ii = 1:d
            R = max(R,sqrt(sum(x(ii).^2)));
        end
    end
    % rou should be negative
    rou = Inf;
    for i = 1:N
        for ii = 1:d
            rou = min(rou,y(ii)*(transpose(w_star)*x(:,ii)));
        end
    end
    t = R^2*(sum(w_star.^2))/(rou^2);
    bounds_minus_ni(I) = abs(num_iters(I) - t);
end
figure;
histogram(num_iters);
xlabel('number of iterations');
ylabel('distribution');

figure;
histogram(transpose(log(bounds_minus_ni))); 
xlabel('log of the difference between the bound and the number of iterations');
ylabel('distribution');
end