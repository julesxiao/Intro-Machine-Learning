function [obbErrorCount] = getPredictions(numBags,tag,tree,X_tr,y_tr)
rowTr=size(X_tr,1);
obbErrorCount = 0;
% for i = 1:rowTr
%     predictionsOobTemp = [];
%     for j = 1:numBags
%         for k = 1:length(tag)
%             tempRow = tag{j}(k);
%             predictionsOob(k,j) = predict(tree{j},X_tr(tempRow,:));
%             predictionsOobTemp = [predictionsOobTemp;predictionsOob(k,j)];  
%         end
%     end
%     predictionsOobError{i} = predictionsOobTemp;
%     if size(predictionsOobError{i}) ~= [0,0]
%         if mode(predictionsOobError{i})~=y_tr(i)
%             obbErrorCount = obbErrorCount+1;
%         end
%     end
% end

for i = 1:rowTr
    predictionsOobTemp = [];
    predictionsOob = [];
    for j = 1:numBags
        if ismember(i,tag{j})==0
            predictionsOob(i,j) = predict(tree{j},X_tr(i,:));
            predictionsOobTemp = [predictionsOobTemp;predictionsOob(i,j)];  
        end
    end
%     predictionsOobError{i} = predictionsOobTemp;
%     if size(predictionsOobError{i},1) ~= 0
    if mode(predictionsOobTemp)~=y_tr(i)
        obbErrorCount = obbErrorCount+1;
    end
%     end
end