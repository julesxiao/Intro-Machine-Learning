function [test_error] = find_test_error(w, X, y)

% find_test_error: compute the test error of a linear classifier w. The
%  hypothesis is assumed to be of the form sign([1 x(n,:)] * w)
%  Inputs:
%		w: weight vector
%       X: data matrix (without an initial column of 1s)
%       y: data labels (plus or minus 1)
%     
%  Outputs:
%        test_error: binary error of w on the data set (X, y) error; 
%        this should be between 0 and 1. 
[r,c1]=size(X);
X_firstcol = ones(r,1);
X = [X_firstcol X];
[datasize,c2] = size(X);
count_error = 0;
for i = 1:datasize
    if 1/(1+exp(-w*transpose(X(i,:)))) >= 0.5
        result = 1;
    else
        result = -1;
    end
    if result~=y(i)
        count_error = count_error + 1;
    end
end
test_error = count_error/datasize;
end

