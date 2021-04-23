function sim = GaussianKernel(x1, x2, sigma)
% Returns a gaussian kernel between x1 and x2
% and returns the value in sim

% Ensure that x1 and x2 are column vectors
x1 = x1(:); x2 = x2(:);


sim = 0;

sim = exp(-sum((x1-x2).^2)/(2*sigma^2));

    
end
