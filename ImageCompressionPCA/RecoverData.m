function X_rec = RecoverData(Z, U, K)
% Recovers an approximation the 
% original data that has been reduced to K dimensions. It returns the
% approximate reconstruction in X_rec.
%

X_rec = zeros(size(Z, 1), size(U, 1));

% Computes the approximation of the data by projecting back
% onto the original space using the top K eigenvectors in U.
%
% For the i-th example Z(i,:), the (approximate)
% recovered data for dimension j is given as follows:
% v = Z(i, :)';
% recovered_j = v' * U(j, 1:K)';
%
               
size(Z)

size(U(:,1: K)')
U_reduce = U(:,1: K)';

 v = Z';

X_rec = v' * U_reduce;

end
