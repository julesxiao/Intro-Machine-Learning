function [w, iterations] = perceptron_learn(data_in)
% perceptron_learn: Run PLA on the input data
% Inputs:  data_in is a matrix with each row representing an (x,y) pair;
%                 the x vector is augmented with a leading 1,
%                 the label, y, is in the last column
% Outputs: w is the learned weight vector; 
%            it should linearly separate the data if it is linearly separable
%          iterations is the number of iterations the algorithm ran for
x = data_in(:,1:11); 
y = data_in(:,12);
% initionlization for w
w = zeros(1, 11);
iterations = 0;
flag = 1;
while flag ~= 0
    % for each training set
    flag = 0;
    for j = 1 : 100
        % randomly pick a misclassified traning example
        if sign(w * transpose(x(j,:))) ~= y(j)
            w_next = w + y(j) * x(j,:);
            w = w_next;
            flag = flag + 1;
        end
    end
    iterations = iterations + 1;
end
end