function Z = ProjectData(X, U, K)
% Computes the projection of the normalized inputs X into the
% reduced dimensional space spanned by
% the first K columns of U. It returns the projected examples in Z.
%

Z = zeros(size(X, 1), K);

% Computes the projection of the data using only the top K 
% eigenvectors in U (first K columns). 
% For the i-th example X(i,:), the projection on to the k-th 
% eigenvector is given as follows: x = X(i, :)';
%                    projection_k = x' * U(:,1: k);

size(X);
size(U);

x = X';

U_reduce = U(:,1: K);
size(U_reduce);

Z =  x'* U_reduce;
size(Z);

end
