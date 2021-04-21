function idx = FindClosestCentroids(X, centroids)
% Returns the closest centroids
% in idx for a dataset X where each row is a single example. idx = m x 1 
% vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% Indexes initialisation
idx = zeros(size(X,1), 1);


for i = 1:length(idx)   % Loop throuth every training example
    min = 0;    
    
    for j =1:K          % Loop through every centroid    
        distance = X(i,:)-centroids(j,:);
        distance = sum(distance.^2);
        size(distance);
        
        if(distance < min || j==1 )
            min = distance;
           idx(i)= j;
        end
       
    end
    
end

size(idx);

end

