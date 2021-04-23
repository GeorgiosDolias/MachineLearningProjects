function CreatePlot(x, y)
% Plots the data points and gives the figure axes labels of
% population and profit.

figure; 


plot(x,y,'bo','MarkerSize',10);
title('Profits vs population of City','FontSize',18);
ylabel('Profits in $10,000s','FontSize',18);
xlabel('Population of City in 10,000s','FontSize',18);
legend
legend('Training data','Location','southeast','FontSize',12,'TextColor','black','LineWidth',1.0);
end