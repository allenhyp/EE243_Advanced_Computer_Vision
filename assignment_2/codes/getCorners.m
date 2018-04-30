function corners = getCorners(I, ncorners)

% I is a 2D matrix 
% ncorners is the number of 'top' corners to be returned
% corners is a ncorners x 2 matrix with the 2D localtions of the corners

% FILL IN YOUR CODE HERE. DO NOT CHANGE ANYTHING ABOVE THIS.


    %% part 0 - Sobel operator
    img=im2double(I);
    dx = [1 2 1; 0 0 0; -1 -2 -1]; % image derivatives
    dy = dx';
    Ix = imfilter(img, dx);    % Step 1: Compute the image derivatives Ix and Iy
    Iy = imfilter(img, dy);
    g = fspecial('gaussian',9,2); % Step 2: Generate Gaussian filter 'g' of size 9x9 and standard deviation Sigma=2.
    Ix2 = imfilter(Ix.^2, g); % Step 3: Smooth the squared image derivatives to obtain Ix2, Iy2 and IxIy
    Iy2 = imfilter(Iy.^2, g);
    IxIy = imfilter(Ix.*Iy, g);

    %% part 1- Compute Matrix E which contains for every point the value
    [r c]=size(Ix2);
    E = zeros(r, c); % Compute matrix E
    for i=2:1:r-1 
        for j=2:1:c-1
         Ix21=sum(sum(Ix2(i-1:i+1,j-1:j+1)));
         Iy21=sum(sum(Iy2(i-1:i+1,j-1:j+1)));
         IxIy1= sum(sum(IxIy(i-1:i+1,j-1:j+1)));

         M=[Ix21 IxIy1;IxIy1 Iy21]; %(1) Build autocorrelation matrix for every singe pixel considering a window of size 3x3
         E(i,j)=min(eig(M)); %(2)Compute Eigen value of the autocorrelation matrix and save the minimum eigenvalue as the desired value.
        end
    end

    %% part 2- Compute Matrix R which contains for every point the cornerness score.
    [r c]=size(Ix2);
    R = zeros(r, c);
    for i=3:1:r-2
        for j=3:1:c-2
         Ix21=sum(sum(Ix2(i-2:i+2,j-2:j+2)));
         Iy21=sum(sum(Iy2(i-2:i+2,j-2:j+2)));
%          IxIy1= sum(sum(IxIy(i-2:i+2,j-2:j+2)));

         R(i,j)=min(Ix21, Iy21);
        end
    end
    %% Part 3 - Select for E and R the 81 most salient points
    % Get the coordinates with maximum cornerness responses
    [sortR,RIX] = sort(R(:),'descend');
    [a, b] = ind2sub([r c],RIX);%The mapping from linear indexes to subscript equivalents for the matrix
    corners = zeros(ncorners*2, 2);
    count = 1;
    figure; imshow(img, []); hold on; % Get the coordinates with maximum cornerness responses     
    for i=1:ncorners
        plot(a(i), b(i), 'r+');
        corners(i,1) = a(i);
        corners(i,2) = b(i);
        count = count + 1;
    end  

    %% Part 4 - Build a function to carry out non-maximal suppression for E and R. Again,the most salient points using a non-maximal suppression of 5x5 pixels.
    R1= ordfilt2(R,25,ones(5));% Get the coordinates with maximum cornerness responses
    R2=(R1==R) & (R > 10);
    [sortR2,R2IX] = sort(R2(:),'descend');
    [a, b] = ind2sub([r c],R2IX); %The mapping from linear indexes to subscript equivalents for the matrix
    figure; imshow(img, []); hold on;   

    for i=1:ncorners
        plot(a(i), b(i), 'r+');
        corners(count+i,1) = a(i);
        corners(count+i,2) = b(i);
    end
