function VisualizeFit(X, mu, sigma2)
% This visualization shows you the 
% probability density function of the Gaussian distribution. Each example
% has a location (x1, x2) that depends on its feature values.


[X1,X2] = meshgrid(0:.5:35); 
Z = multivariateGaussian([X1(:) X2(:)],mu,sigma2);
Z = reshape(Z,size(X1));

plot(X(:, 1), X(:, 2),'bx');
legend('Dataset points','FontSize',12,'TextColor','black','LineWidth',1.0);

hold on;
% Do not plot if there are infinities
if (sum(isinf(Z)) == 0)
    contour(X1, X2, Z, 10.^(-20:3:0)','g','DisplayName','Fit');
end
hold off;

end