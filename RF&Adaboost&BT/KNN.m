x = [2,0;5,2;6,3;0,1;2,3;4,4];
y = [1;1;1;-1;-1;-1];
labels = categories(categorical(y));
Mdl = fitcknn(x,y,'NumNeighbors',1,'Standardize',1);
%x1range = min(x(:,1)):.01:max(x(:,1));
x1range = min(x(:,1)):.01:6;
x2range = min(x(:,2)):.01:max(x(:,2));
[xx1, xx2] = meshgrid(x1range,x2range);
XGrid = [xx1(:) xx2(:)];
predictedspecies = predict(Mdl,XGrid);
figure(1);
gscatter(xx1(:), xx2(:), predictedspecies,'rg');
legend off, axis tight
legend(labels,'Location',[0.35,0.01,0.35,0.05],'Orientation','Horizontal');

x_5 = [2,0;5,10;6,15;0,5;2,15;4,20];
% y = [1;1;1;-1;-1;-1];
% labels = categories(categorical(y));
%Mdl_5 = fitcknn(x_5,y,'NumNeighbors',1,'Standardize',1);
Mdl_5 = fitcknn(x_5,y,'NumNeighbors',1);
%x1range_5 = min(x_5(:,1)):.01:max(x_5(:,1));
x1range_5 = -1:.01:12;
x2range_5 = min(x_5(:,2)):.01:max(x_5(:,2));
[xx1_5, xx2_5] = meshgrid(x1range_5,x2range_5);
XGrid_5 = [xx1_5(:) xx2_5(:)];
predictedspecies_5 = predict(Mdl_5,XGrid_5);
figure(2);
gscatter(xx1_5(:), xx2_5(:), predictedspecies_5,'rg');
legend off, axis tight
legend(labels,'Location',[0.35,0.01,0.35,0.05],'Orientation','Horizontal');
