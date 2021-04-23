function p = PredictOneVsAll(all_theta, X)
%  Predicts the label for a trained one-vs-all classifier. The labels 
%  are in the range 1..K, where K = size(all_theta, 1). 
%  It will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];
      

h = X*all_theta';
[M I]= max(h, [],2);

p=I;

end
