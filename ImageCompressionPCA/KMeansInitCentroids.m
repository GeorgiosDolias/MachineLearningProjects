function centroids = KMeansInitCentroids(X, K)
% Returns K initial centroids to be
% used with the K-Means on the dataset X


centroids = zeros(K, size(X, 2));


% Randomly reorders the indices of examples
randidx = randperm(size(X,1));

% Takes the first K examples as centroid
centroids = X(randidx(1:K),:);

end

