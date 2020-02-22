function [t, w, e_in] = logistic_reg(X, y, w_init, max_its, eta)

% logistic_reg: learn a logistic regression model using gradient descent
%  Inputs:
%       X:       data matrix (without an initial column of 1s)
%       y:       data labels (plus or minus 1)
%       w_init:  initial value of the w vector (d+1 dimensional)
%       max_its: maximum number of iterations to run for
%       eta:     learning rate
%     
%  Outputs:
%        t:    the number of iterations gradient descent ran for
%        w:    learned weight vector
%        e_in: in-sample (cross-entropy) error 

% add an initial column of 1s to X
[r,c]=size(X);
X_firstcol = ones(r,1);
X = [X_firstcol X];
[datasize,c]=size(X);
epsilon = 10^(-3);
if max_its == Inf
    epsilon = 10^(-6);
end
% max_its = 10^5;
% max_its = 10^6;
iterations = 0;

gradient = 0;
while (iterations < max_its)
    % for each training set
    % find the largest magnitude of all the elements of the gradient
%     max = -10000;
    for i = 1:datasize
        gradient_i = -y(i)*X(i,:)/(exp(y(i)*(w_init)*transpose(X(i,:)))+1);
        gradient = gradient + gradient_i;
    end
    gradient = (1/datasize)*gradient;
    count = 0;
    for i = 1:c
        if abs(gradient(i)) < epsilon
            count = count+1;
        end
    end
    if count == c
        break;
    end
    w_init = w_init - eta * gradient;
    iterations = iterations + 1;
end
e_in = 0;
w = w_init;
for i = 1:datasize
    e_in = e_in + log(1 + exp(-y(i)*w*transpose(X(i,:))));
end
e_in = (1/datasize)*e_in;

t = iterations;

