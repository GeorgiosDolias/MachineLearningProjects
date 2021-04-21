function J = computeCost(X, y, theta)

% Initialize some useful values
m = length(y);
J = 0;
sum = 0;

h=X*theta;

for i=1:m
    sum= sum + (h(i)-y(i))^2;
end
    
J = (1/(2*m))*sum;

end
