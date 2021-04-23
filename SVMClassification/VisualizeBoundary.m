function VisualizeBoundary(X, y, model,dataset)
% Plots a non-linear decision 
% boundary learned by the SVM and overlays the data on it

% Create New Figure
figure; hold on;

% Finds Indices of Positive and Negative Examples
pos = find(y == 1); neg = find(y == 0);

% Plots Examples
plot(X(pos, 1), X(pos, 2), 'k+','LineWidth', 2, 'MarkerSize', 7);

plot(X(neg, 1), X(neg, 2), 'ko', 'MarkerFaceColor', 'c', 'MarkerSize', 7);

xlabel('First dimension','FontSize',18)
ylabel('Second dimension','FontSize',18)

title(['Decision boundary of: ', dataset],'FontSize',18)
legend('y = 1', 'y = 0','FontSize',12,'TextColor','blue','LineWidth',1.0)

% Makes classification predictions over a grid of values
x1plot = linspace(min(X(:,1)), max(X(:,1)), 100)';
x2plot = linspace(min(X(:,2)), max(X(:,2)), 100)';
[X1, X2] = meshgrid(x1plot, x2plot);
vals = zeros(size(X1));
for i = 1:size(X1, 2)
   this_X = [X1(:, i), X2(:, i)];
   vals(:, i) = SvmPredict(model, this_X);
end

% Plot the SVM boundary
hold on
contour(X1, X2, vals, [0.5 0.5], 'b','DisplayName','boundary');
%hblue = plot(NaN, 'b');
%legend(hblue,'b');
hold off;

end
