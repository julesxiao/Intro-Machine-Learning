function [oob_err, test_err] = BaggedTrees(X_tr, y_tr, X_te, y_te, numBags)
% BaggedTrees: Learns an ensemble of numBags CART decision trees on the input dataset 
%              and also plots the out-of-bag error as a function of the number of bags
%      Inputs:
%              X_tr: Training data
%              y_tr: Training labels
%              X_te: Testing data
%              y_te: Testing labels
%              numBags: Number of trees to learn in the ensemble
%     Outputs: 
%	           oob_err: Out-of-bag classification error of the final learned ensemble
%              test_err: Classification error of the final learned ensemble on test data
%
% You may use "fitctree" but not "TreeBagger" or any other inbuilt bagging function
[rowTr,cTr]=size(X_tr);
[rowTe,cTe]=size(X_te);
[tag,tree] = CreateEnsemble(X_tr, y_tr, numBags);
%% Calculate Out-of-bag classification error of the final learned ensemble 
predictionsOob = zeros(rowTr,numBags);

% get predictions for all uncounted trees for all bags
for i = 1:rowTr
    for j = 1:numBags
        if ismember(i,tag{j}) == 1
            predictionsOob(i,j) = predict(tree{j},X_tr(i,:));
        end
    end
end

% get data for the plot
for bag = 1:numBags
    obbErrorCount = 0;
    for i = 1:rowTr
        predictionsOobTemp = [];
        for j = 1:bag
            if ismember(i,tag{j}) == 1
                predictionsOobTemp = [predictionsOobTemp;predictionsOob(i,j)];  
            end
        end
        if size(predictionsOobTemp,1) ~= 0
            if mode(predictionsOobTemp)~=y_tr(i)
                    obbErrorCount = obbErrorCount+1;
            end 
        end
    end
    oob_error(bag) = obbErrorCount/rowTr;
%     disp(bag);
%     disp(oob_error(bag));
end
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

% plot(oob_error);
% xlabel('number of bags');
% ylabel('oob error');
end
