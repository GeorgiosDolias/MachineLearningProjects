function centroids = KMeansInitCentroids(X, K)
% Returns K initial centroids to be
% used with the K-Means on the dataset X


% You should return this values correctly
centroids = zeros(K, size(X, 2));


% Randomly reorder the indices of examples
randidx = randperm(size(X,1));

% Take the first K examples as centroid
centroids = X(randidx(1:K),:);

end

