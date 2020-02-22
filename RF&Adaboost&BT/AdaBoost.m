function [train_err, test_err] = AdaBoost(X_tr, y_tr, X_te, y_te, numTrees)
% AdaBoost: Implement AdaBoost with decision stumps as the weak learners. 
%           (hint: read the "Name-Value Pair Arguments" part of the "fitctree" documentation)
%   Inputs:
%           X_tr: Training data
%           y_tr: Training labels
%           X_te: Testing data
%           y_te: Testing labels
%           numTrees: The number of trees to use
%  Outputs: 
%           train_err: Classification error of the learned ensemble on the training data
%           test_err: Classification error of the learned ensemble on test data
% 
% You may use "fitctree" but not any inbuilt boosting function

%initialize input weights

n = size(y_tr,1);
w = ones(n,1)/n;
alpha_set = zeros(numTrees,1);
%initialize hypothesis
h_set = {};
mean = (min(y_tr)+max(y_tr))/2;

for nm = 1:size(y_tr,1)
    y_tr(nm) = y_tr(nm)-mean;
end
for nm = 1:size(y_te,1)
    y_te(nm) = y_te(nm)-mean;
end


for i = 1:numTrees
    %Train a weak learner, ht, by minimizing the weighted training error of ht 
    weaker_learner = fitctree(X_tr, y_tr, 'MaxNumSplits',1, 'Weights', w,'SplitCriterion', 'deviance');
    h_set{i} = weaker_learner;
    %compute the weighted trainning error of ht
    y_weaker = predict(weaker_learner,X_tr);
    epsilon = 0;
    for j = 1:n
        if (y_weaker(j) ~= y_tr(j))
            epsilon = epsilon + w(j);
        end
    end
    %compute the "importance" of ht:
    alpha = 0.5*log((1-epsilon)/epsilon);
    alpha_set(i) = alpha;
    %update the weight
    
    for j = 1:n
        if y_weaker(j) ~= y_tr(j)
            w(j) = w(j) * exp(alpha);
        else
            w(j) = w(j) * exp(-alpha);
        end
    end
    zt = 2* sqrt(epsilon*(1-epsilon));
    w = w/zt;
    
end
%output an aggregated hypothesis
train_error_set = zeros(numTrees,1);

for T = 1:numTrees
    y = zeros(n,1);
    for t = 1:T
%         vhvh=predict(h_set{t},X_tr);
        y = y + alpha_set(t).*predict(h_set{t},X_tr);
    end
    sum_err = 0;
    for index = 1:n
       
        if sign(y(index))~=(y_tr(index))
            sum_err = sum_err+1;
        end
    end
    train_error_set(T)=sum_err/n;
end

test_error_set = zeros(numTrees,1);


for T = 1:numTrees
    y = zeros(size(y_te,1),1);
    for t = 1:T
        y = y + alpha_set(t).*predict(h_set{t},X_te);
        
    end

    sum_err = 0;
    for index = 1:size(y_te,1)
        if sign(y(index))~=(y_te(index))
            sum_err = sum_err+1;
        end
    end
    test_error_set(T)=sum_err/size(y_te,1);
end


train_err = train_error_set;
test_err = test_error_set;