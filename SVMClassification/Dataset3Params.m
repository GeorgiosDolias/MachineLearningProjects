function [C, sigma] = Dataset3Params(X, y, Xval, yval)
% Returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel


C = 0.01;
sigma = 0.01;


C_values = [0.01 0.03 0.1 0.3 1 3 10 30]';

sigma_values = [0.01 0.03 0.1 0.3 1 3 10 30]';

error_max = 1;
error = 0;

for i=1:length(C_values)
    for j=1:length(sigma_values)
        model= svmTrain(X, y, C_values(i), @(x1, x2) gaussianKernel(x1, x2, sigma_values(j)));
        predictions = svmPredict(model, Xval);
        error = mean(double(predictions ~= yval));
        
        if error < error_max;
           C = C_values(i);
           sigma = sigma_values(j);
           error_max = error
        end
    end
end

end
