function PlotData(X, y)

%   Plots the data points with + for the positive (accepted) examples
%   and x for the negative (rejected) examples. X is assumed to be a Mx2 matrix.

%   Create New Figure
figure; hold on;

%   Find indices of Positive and negative examples
pos = find(y==1); neg = find(y==0);

%   Plot Examples
plot(X(pos,1),X(pos,2),'k+','LineWidth',2,'MarkerEdgeColor',"g",...
    'MarkerSize',7);
plot(X(neg,1),X(neg,2),'kx','LineWidth',2,'MarkerFaceColor',[1,0,0],...
    'MarkerSize',7);

hold off;

end