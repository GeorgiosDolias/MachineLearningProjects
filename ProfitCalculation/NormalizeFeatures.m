function [X_norm, mu, sigma] = NormalizeFeatures(X)
% Returns a normalized version of X where
% the mean value of each feature is 0 and the standard deviation
% is 1.


X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));
 

for i=1:size(X, 2)
   
   mean1 = mean(X(:,i));
   X_norm(:,i)= X_norm(:,i)-mean1;
   mu(i)= mean(X_norm(:,i));
   sig = std(X_norm(:,i));
   X_norm(:,i)= X_norm(:,i)/sig;
   sigma(i) = std(X_norm(:,i));
end
   

end
