function PlotsData(X, y,dataset)
% Plots the data points with + for the positive examples
% and o for the negative examples. X is assumed to be a Mx2 matrix.
%

% Create New Figure
figure; hold on;

% Find Indices of Positive and Negative Examples
pos = find(y == 1); neg = find(y == 0);

% Plot Examples
plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2, 'MarkerSize', 7);

plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'c', 'MarkerSize', 7);

xlabel('First dimension','FontSize',18)
ylabel('Second dimension','FontSize',18)

title(['Visualisation of ', dataset],'FontSize',18)
legend('y = 1', 'y = 0','FontSize',12,'TextColor','blue','LineWidth',1.0)
hold off;

end
