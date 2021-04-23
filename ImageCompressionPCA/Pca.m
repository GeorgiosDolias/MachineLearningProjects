function [U, S] = Pca(X)

% Returns the eigenvectors U, the eigenvalues (on diagonal) in S

% Useful values
[m, n] = size(X);

U = zeros(n);
S = zeros(n);

S = (X'*X)/m;

% Computes the eigenvectors and eigenvalues of the covariance matrix. 
[U, S, V] = svd(S);


end
