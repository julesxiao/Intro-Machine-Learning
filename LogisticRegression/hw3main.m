clear all
close all

%% getting the training set and the test set
% 152 * 14
Train = table2array(readtable('cleveland_train.csv'));
% 145 * 14, the 14th column is y.
Test = table2array(readtable('cleveland_test.csv'));

[TrainNumRows,TrainNumCols] = size(Train);
[TestNumRows,TestNumCols] = size(Test);
% change the labels to -1 if they'e 0
for i = 1: TrainNumRows
    if Train(i,TrainNumCols) == 0
        Train(i,TrainNumCols) = -1;
    end
end

for i = 1: TestNumRows
    if Test(i,TestNumCols) == 0
        Test(i,TestNumCols) = -1;
    end
end

%% Normalization
TrainNorm = Train;
for i = 1: TrainNumCols-1
    meansum = 0;
    devsum = 0;
    for j = 1:TrainNumRows
        meansum = meansum + Train(j,i);
    end
    mean(i) = meansum/TrainNumRows;
    for j = 1:TrainNumRows
        devsum = devsum+ (Train(j,i) - mean(i))^2;
    end
    dev(i) = sqrt(devsum/TrainNumRows);
    for j = 1:TrainNumRows     
        TrainNorm(j,i) = (Train(j,i) - mean(i))/dev(i);
    end
end

% Normalize the training data outputs, save the mean and variance and use
% those to normalize the test data outputs
TestNorm = Test;
for i = 1: TestNumCols-1
    for j = 1:TestNumRows
        TestNorm(j,i) = (Test(j,i) - mean(i))/dev(i);
    end
end

w_init = zeros(1, TrainNumCols);
eta = 10^(-5);

%%%% comment this part examine part b
%% Problem(a)i
%% Model 1 
max_its = 10^4;
tic;
[t1, w1, e_in1] = logistic_reg(Train(:,1:TrainNumCols-1), Train(:,TrainNumCols), w_init, max_its, eta);
time1 = toc;
[test_error_training1] = find_test_error(w1, Train(:,1:TrainNumCols-1), Train(:,TrainNumCols));
[test_error_test1] = find_test_error(w1, Test(:,1:TrainNumCols-1), Test(:,TrainNumCols));
% compute the test error on both sets

%% Model 2
max_its = 10^5;
tic;
[t2, w2, e_in2]=logistic_reg(Train(:,1:TrainNumCols-1), Train(:,TrainNumCols), w_init, max_its, eta);
time2 = toc;
[test_error_training2] = find_test_error(w2, Train(:,1:TrainNumCols-1), Train(:,TrainNumCols));
[test_error_test2] = find_test_error(w2, Test(:,1:TrainNumCols-1), Test(:,TrainNumCols));
%% Model 3
max_its = 10^6;
tic;
[t3, w3, e_in3]=logistic_reg(Train(:,1:TrainNumCols-1), Train(:,TrainNumCols), w_init, max_its, eta);
time3 = toc;
[test_error_training3] = find_test_error(w3, Train(:,1:TrainNumCols-1), Train(:,TrainNumCols));
[test_error_test3] = find_test_error(w3, Test(:,1:TrainNumCols-1), Test(:,TrainNumCols));
%%%%end comment
%% Problem(b)
Eta = [0.01, 0.1, 1, 4, 5, 6, 7, 7.5, 7.6, 7.7];
[r, num]=size(Eta);
Ein = zeros(1,num);
Iterations = zeros(1,num);
% elapsed time
time = zeros(1,num);
TestError = zeros(1,num);
% no maximum number of iterations on problem(b)
max_its = Inf;
for i = 1: num
    eta = Eta(i);
    tic
    [t, w, e_in] = logistic_reg(TrainNorm(:,1:TrainNumCols-1), TrainNorm(:,TrainNumCols), w_init, Inf,eta);
    time(i) = toc;
    Ein(i) = e_in;
    Iterations(i) = t;
    [test_error] = find_test_error(w,TestNorm(:,1:TestNumCols-1),TestNorm(:,TestNumCols));
    TestError(i) = test_error;
end
