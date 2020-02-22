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
%X_tr_new = zeros(rowTr,cTr);
%rng('default');
%A = randi(rowTr,1,rowTr);
%X_tr_new = zeros([rowTr,cTr,numBags]);
%y_tr_new = zeros([rowTr,cTr]);
mark = 1;
% tag = zeros([rowTr,numBags]);
% for i = 1:numBags
%     %sample from X_tr to create a dataset with replacement 
%     for j = 1:rowTr
%         index = randi(rowTr);
%         X_tr_new(j,:,i) = X_tr(index,:);
%         y_tr_new(j,:,i) = y_tr(index);
%     %         X_tr_temp = zeros([1,cTr]);
%     %         X_tr_temp(1,:) = X_tr(A(1,1),:);
%     %         y_tr_temp(1,:) = y_tr(A(1,1),:);
%     %     for j = 2:rowTr
%     %         X_tr_temp = [X_tr_temp;X_tr(A(1,j),:)];
%     %         y_tr_temp = [y_tr_temp;y_tr(A(1,j),:)];
%     %     end
%     % mark trees which were trained
%         tag(index,i) = mark;
%     end
% end
% [X_tr_new, y_tr_new,tag,tree] = createEnsemble(X_tr, y_tr, X_te, y_te, numBags);
% %Learns an ensemble of numBags CART decision trees
% for i = 1:numBags 
%     tree{i} = fitctree(X_tr_new(:,:,i), y_tr_new(:,:,i));
%     %tree{i} = fitctree(X_tr_new(:,:,i), y_tr_new(:,:,i),'NumVariablesToSample','all');
% end
%% Calculate Out-of-bag classification error of the final learned ensemble 
% find out trees which were not used to train
% get aggregated prediction for each x
% predictionsOob = zeros(rowTr,numBags);
% for i = 1:rowTr
%     predictionsOobTemp = [];
%     for j = 1:numBags
%         if tag(i,j)~=mark
%             predictionsOob(i,j) = predict(tree{j},X_tr(i,:));
%             predictionsOobTemp = [predictionsOobTemp;predictionsOob(i,j)];  
%         end
%     end
%     predictionsOobError{i} = predictionsOobTemp;
% end
%[predictionsOobError] = getPredictions(numBags,tag,tree,X_tr);
oob_error = zeros([1,numBags]);
for i = 1:numBags
    [tag,tree] = CreateEnsemble(X_tr, y_tr, X_te, y_te, i);
    [predictionsOobError] = getPredictions(i,tag,tree,X_tr);
    obbErrorCount = 0;
    for j = 1:rowTr
        if mode(predictionsOobError{j})~=y_tr(j)
            obbErrorCount = obbErrorCount+1;
        end   
    end
    oob_error(i) = obbErrorCount/rowTr;
    disp(i);
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

% for j = 1:numBags
%     testErrors(j) = sum(predict(tree{j},X_te)~= y_te)/length(y_te);
% end
% test_err = sum(testErrors)/numBags;
plot(oob_error);
xlabel('Number of bags');
ylabel('Out of bag error');

