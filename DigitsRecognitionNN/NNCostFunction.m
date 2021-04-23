function [J grad] = NNCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
% Implements the neural network cost function for a two layer
% neural network which performs classification
% Computes the cost and gradient of the neural network. The
% parameters for the neural network are "unrolled" into the vector
% nn_params and need to be converted back into the weight matrices. 
% 
% The returned parameter grad should be an "unrolled" vector of the
% partial derivatives of the neural network.
%

% Reshapes nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
K = num_labels;
X= [ones(m, 1) X];  

J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));


% Part 1: Feedforwards the neural network and returns the cost in the
%         variable J.
%
% Part 2: Implements the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. 
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. This vector is mapped into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
% Part 3: Implements regularization with the cost function and gradients.
%
%

%   Step 1 
all_combos = eye(num_labels);    
y_matrix = all_combos(y,:) ; 
size(y_matrix);

%   Step 2
a1=X;

a2 = Sigmoid(a1 * Theta1');

h = Sigmoid([ones(m, 1) a2] * Theta2');



%   Step 3a-> Summations for non-regularized cost function

sumi=0;
for i=1:m
    sumk=0;
    
    for k=1:K
        sumk = sumk + (-y_matrix(i,k)*log(h(i,k))-(1-y_matrix(i,k))*log(1-h(i,k)));        
    end
    
    % Vectorized cost function calculation    
    sumi= sumi + sumk;
end

%Step 3b-> Summations for Theta

%Sum for Theta1
sumj1=0;

for j=1:size(Theta1, 1)
    sumj1 = sumj1 + sum(Theta1(j,2:end).^2); 
end


%Sum for Theta2
sumj2=0;
for j=1:size(Theta2, 1)
    sumj2 = sumj2 + sum(Theta2(j,2:end).^2); 
end


%Step 3c-> Regularized cost function

J = (1/m)*sumi+(lambda/(2*m))*(sumj1+sumj2);


% Step 4 Backpropagation algorithm

D1= zeros(size(Theta1));
D2=zeros(size(Theta2));

for t=1:m
    
    a1=X(t,:);
    size(a1);
    
    z2=a1 * Theta1';
    a2 = sigmoid(z2);
    size(a2);
    a2 = [ones(size(a2,1), 1) a2];
    size(a2);
    
    z3 = a2* Theta2';    
    a3 = sigmoid(z3);
    
    size(a3);
    
    
    d3 = h(t,:)-y_matrix(t,:);
    size(d3);
    size(Theta2(:,2:end)');
    
    d2 =(d3*Theta2(:,2:end)).*sigmoidGradient(z2);
    size(d2);
   
    size(d3'*a2);
    
    D2 =D2 + d3'*a2;
    size(D2);
    D1 =D1 + d2'*a1;
    size(D1);
end


Theta1_grad = (1/m)*D1;
Theta2_grad = (1/m)*D2;

Theta1_grad(:,2:end) =Theta1_grad(:,2:end) + (lambda/m)*Theta1(:,2:end);
Theta2_grad(:,2:end) =Theta2_grad(:,2:end) + (lambda/m)*Theta2(:,2:end);


% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
