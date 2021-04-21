function PlotDecisionBoundary(theta, X, y)
% Plots the data points with + for the 
% positive examples and o for the negative examples. X is assumed to be 
% a either 
% 1) Mx3 matrix, where the first column is an all-ones column for the 
%    intercept.
% 2) MxN, N>3 matrix, where the first column is all-ones

% Plots Data
PlotData(X(:,2:3), y);
hold on

if size(X, 2) <= 3
    % Only need 2 points to define a line, so choose two endpoints
    plot_x = [min(X(:,2))-2,  max(X(:,2))+2];

    % Calculates the decision boundary line
    plot_y = (-1./theta(3)).*(theta(2).*plot_x + theta(1));

    % Plots, and adjusts axes for better viewing
    plot(plot_x, plot_y)
    
    % Legend, specific for the exercise
    legend('Admitted', 'Not admitted', 'Decision Boundary','FontSize',18)
    axis([30, 100, 30, 100])
else
    % Here is the grid range
    u = linspace(-1, 1.5, 50);
    v = linspace(-1, 1.5, 50);

    z = zeros(length(u), length(v));
    % Evaluate z = theta*x over the grid
    for i = 1:length(u)
        for j = 1:length(v)
            z(i,j) = mapFeature(u(i), v(j))*theta;
        end
    end
    z = z'; % important to transpose z before calling contour

    % Plot z = 0
    
    contour(u, v, z, [0, 0], 'LineWidth', 2)
end
hold off

end