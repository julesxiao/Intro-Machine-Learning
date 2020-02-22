%generate x1
x1 = 2*rand([1 6]) - 1;

% 3 data points [0,1]
data_upper_x2 = rand([1 3]);
% 3 data points [-1,0]
data_lower_x2 = rand([1 3])-1;

%generate data points
x2 = [data_lower_x2,data_upper_x2];

%compute thresholds a_random
a_random = (min(data_upper_x2) - max(data_lower_x2))*rand + max(data_lower_x2);

%compute threshold a_svm
a_svm = (min(data_upper_x2) + max(data_lower_x2))/2;

%plot g_random
figure()

scatter(x1(1:3),data_upper_x2,[],'r','filled');

hold on;
scatter(x1(4:6),data_lower_x2,[],'g','filled');

%plot sign(x2-a_random);
yline(a_random,'b-');

yline(a_svm,'c-');

hold off;
legend('classed as +1', 'classed as -1','aRandom for gRandom','aSVM for gSVM','Location','northeastoutside','FontSize',24);
xlim([-1 1]);
ylim([-1 1]);
grid on;
xlabel('x1');
ylabel('x2');
