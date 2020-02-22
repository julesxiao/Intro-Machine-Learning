function [tag,tree] = CreateEnsembleRF(X_tr, y_tr, numBags,m)
% mark = 1;
rowTr=size(X_tr,1);
for i = 1:numBags
    pick = randsample(rowTr,rowTr,true);
    X_tr_new{i} = X_tr(pick,:);
    y_tr_new{i} = y_tr(pick,:);
    %record the rows that were not used to train
    tag{i} = setdiff([1:rowTr],transpose(pick));
    %Learns an ensemble of numBags CART decision trees
    tree{i} = fitctree(X_tr_new{i}, y_tr_new{i},'NumVariablesToSample',m);
end