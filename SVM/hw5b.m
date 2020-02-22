iterations = 1000000;
aRandom = zeros(1,iterations);
aSvm = zeros(1,iterations);
for i = 1:iterations
    %generate x1
    x1 = 2*rand([1 6]) - 1;

    % 3 data points [0,1]
    data_upper_x2 = rand([1 3]);
    % 3 data points [-1,0]
    data_lower_x2 = rand([1 3])-1;

    %generate data points
    x2 = [data_lower_x2,data_upper_x2];

    %compute thresholds a_random
    aRandom(i) = (min(data_upper_x2) - max(data_lower_x2))*rand + max(data_lower_x2);

    %compute threshold a_svm
    aSvm(i) = (min(data_upper_x2) + max(data_lower_x2))/2;
    disp(i);
end

%plot g_random
figure();
edges = linspace(-1, 1, 41); % Create 20 bins.
histogram(aRandom, 'BinEdges',edges);
xlabel('aRandom values');
ylabel('Histogram of learned values for aRandom');
xlim([-1 1]);

figure();
histogram(aSvm, 'BinEdges',edges);
xlabel('aSVM values');
ylabel('Histogram of learned values for aSVM');
xlim([-1 1]);