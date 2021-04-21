function PlotData(X, y)
% Plots the data points with + for the positive examples
% and o for the negative examples. X is assumed to be a Mx2 matrix.


figure; hold on;


% Find indices of Positive and negative examples
pos = find(y==1); neg = find(y==0);

% Plot Examples
plot(X(pos,1),X(pos,2),'k+','LineWidth',2,...
    'MarkerSize',7);
plot(X(neg,1),X(neg,2),'ko','MarkerFaceColor','c',...
    'MarkerSize',7);






% =========================================================================



hold off;

end
