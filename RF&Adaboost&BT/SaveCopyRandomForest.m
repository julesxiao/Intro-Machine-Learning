function [oob_err, test_err] = RandomForest(X_tr, y_tr, X_te, y_te, numBags, m)
% RandomForest: Learns an ensemble of numBags CART decision trees using a random subset of
%               the features at each split on the input dataset and also plots the 
%               out-of-bag error as a function of the number of bags
%       Inputs:
%               X_tr: Training data
%               y_tr: Training labels
%               X_te: Testing data
%               y_te: Testing labels
%               numBags: Number of trees to learn in the ensemble
% 				m: Number of randomly selected features to consider at each split
% 				   (hint: read the "Name-Value Pair Arguments" part of the fitctree documentation)
%      Outputs: 
%	            oob_err: Out-of-bag classification error of the final learned ensemble
%               test_err: Classification error of the final learned ensemble on test data
%
% You may use "fitctree" but not "TreeBagger" or any other inbuilt bagging function
[rowTr,cTr]=size(X_tr);
[rowTe,cTe]=size(X_te);

mark = 1;
tag = zeros([rowTr,numBags]);
for i = 1:numBags
    %sample from X_tr to create a dataset with replacement 
    for j = 1:rowTr
        index = randi(rowTr);
        X_tr_new(j,:,i) = X_tr(index,:);
        y_tr_new(j,:,i) = y_tr(index);
        tag(index,i) = mark;
    end
end

%Learns an ensemble of numBags CART decision trees
for i = 1:numBags 
    % Not sure
    tree{i} = fitctree(X_tr_new(:,:,i), y_tr_new(:,:,i),'NumVariablesToSample',m);
end
%% Calculate Out-of-bag classification error of the final learned ensemble 
% find out trees which were not used to train
% get aggregated prediction for each x
predictionsOob = zeros(rowTr,numBags);
for i = 1:rowTr
    predictionsOobTemp = [];
    for j = 1:numBags
        if tag(i,j)~=mark
            predictionsOob(i,j) = predict(tree{j},X_tr(i,:));
            predictionsOobTemp = [predictionsOobTemp;predictionsOob(i,j)];  
        end
    end
    predictionsOobError{i} = predictionsOobTemp;
end

for i = 1:numBags
    obbErrorCount = 0;
    for j = 1:rowTr
        if mode(predictionsOobError{j})~=y_tr(j)
            obbErrorCount = obbErrorCount+1;
        end   
    end
    oob_error(i) = obbErrorCount/rowTr;
end

% get the aggregated prediction
oob_err = oob_error(numBags);

%% calculate the Classification error of the final learned ensemble on test data
testErrorCount = 0;
for i = 1:rowTe
    for j = 1:numBags
        predictions(j) = predict(tree{j},X_te(i,:));
    end
    % aggregate the predictions of each tree on each point
    if mode(predictions)~=y_te(i)
        testErrorCount = testErrorCount+1;
    end
end
test_err = testErrorCount/rowTe;
end
