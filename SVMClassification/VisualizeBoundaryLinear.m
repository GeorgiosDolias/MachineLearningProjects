function VisualizeBoundaryLinear(X, y, model,dataset)
% Plots a linear decision boundary 
% learned by the SVM and overlays the data on it

w = model.w;
b = model.b;
xp = linspace(min(X(:,1)), max(X(:,1)), 100);
yp = - (w(1)*xp + b)/w(2);
% Create New Figure
figure; hold on;

% Find Indices of Positive and Negative Examples
pos = find(y == 1); neg = find(y == 0);

% Plot Examples
plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2, 'MarkerSize', 7);

plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'c', 'MarkerSize', 7);

xlabel('First dimension','FontSize',18)
ylabel('Second dimension','FontSize',18)

title(['Decision boundary of ', dataset],'FontSize',18)

hold on;
plot(xp, yp, '-b'); 
legend('y = 1', 'y = 0','Boundary','FontSize',12,'TextColor','blue','LineWidth',1.0)
hold off

end
