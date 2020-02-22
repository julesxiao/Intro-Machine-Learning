figure;
x = linspace(0, 1);
y = 4*exp(-12*x.^2); 
x1 = linspace(0, 0.1667);
x2 = linspace(0.1667, 0.3333);
x3 = linspace(0.3333, 0.5);
x4 = linspace(0.5,1);
xii = [x1,x2,x3,x4];
z = 0.9023 * ones(1, length(x1));
i = 0.4856 * ones(1, length(x2));
j = 0.0615 * ones(1, length(x3));
k = 0 * ones(1, length(x4));
yii = [z,i,j,k];

plot(x,y);
hold on
%plot(0:0.1667,z,'b', 0.1667:0.3333,i, 0.3333:0.5, j, 0.5:1,k,'Color','r');
plot(xii,yii,'Color','r');
xlabel('\epsilon')
ylabel('Probability')
hold off
legend({'Hoeffding Bound', 'Computational Probability'},'location','northeast')
