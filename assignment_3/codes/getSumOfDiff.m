function sod = getSumOfDiff(I)

% I is a 3D tensor of image sequence where the 3rd dimention represents the time axis

% YOUR CODE HERE. DO NOT CHANGE ANYTHING ABOVE THIS.
%     figure;
%     subplot(3,1,1); imshow(I(:,:,1));
%     subplot(3,1,2); imshow(I(:,:,2));
%     subplot(3,1,3); imshow(I(:,:,3));
    [m, n, N] = size(I);
    sod = zeros(m, n);
    for i = 1:N-1
        for j = i:N
            sod(:,:) = abs(I(:,:,i)-I(:,:,j))/(N*(N-1)/2);
        end
    end
    
end