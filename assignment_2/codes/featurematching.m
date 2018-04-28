% DO NOT MODIFY ANYTHING IN THIS CODE EXCEPT numpoints (if required for analysis)

clear all; close all; clc
addpath(genpath('vlfeat'));

I = imread('blocks.png');  % Original image
theta = 30*pi/180;
tform1 = affine2d([cos(theta) -sin(theta) 0; sin(theta) cos(theta) 0; 0 0 1]);
tform2 = affine2d([1 0 0; 0.5 1 0; 0 0 1]);
R = imwarp(I,tform1);      % Transformed image
figure,
subplot(1,2,1),imshow(I); title('Original Image')
subplot(1,2,2),imshow(R); title('Transformed Image')

% Corner extraction
numpoints = 75;
cornersI = getCorners(I,numpoints); 
cornersR = getCorners(R,numpoints); 
figure,title('Original Image Corners');
subplot(1,2,1),imshow(I); hold on; plot(cornersI(:,2),cornersI(:,1),'*r'); title('Original Image Corners');
subplot(1,2,2),imshow(R); hold on; plot(cornersR(:,2),cornersR(:,1),'*r'); title('Transformed Image Corners');
% HoG feature extraction
featI = getFeatures(I,cornersI);
featR = getFeatures(R,cornersR);

% Obtaining matches
matches = getMatches(featI,featR);

coordinatesI = cornersI(matches(:,1),:);
coordinatesR = cornersR(matches(:,2),:);
figure,showMatchedFeatures(I,R,flip(coordinatesI,2),flip(coordinatesR,2),'montage'), title('HoG matching')

% SIFT Feature extraction and matching
[framesI, descI] = getSIFTFeatures(I);
[framesR, descR] = getSIFTFeatures(R);
siftmatches = getMatches(descI',descR');
siftcoordinatesI = framesI(1:2,siftmatches(:,1))';
siftcoordinatesR = framesR(1:2,siftmatches(:,2))';
figure,showMatchedFeatures(I,R,siftcoordinatesI,siftcoordinatesR,'montage'), title('SIFT matching')

