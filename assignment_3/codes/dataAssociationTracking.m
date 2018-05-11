% DO NOT CHANGE THIS CODE

clear all; close all;

v = VideoReader('atrium.mp4');
I = read(v);
for i = 1:size(I,4)
    J(:,:,i) = rgb2gray(I(:,:,:,i));
end

offset = 50;
D1 = getSumOfDiff(J(:,:,offset-1:offset+1));
bbox1 = getDetections(D1);
feat1 = getFeatures(J(:,:,offset),bbox1);

for i = offset:(size(J,3)-2)

    D2 = getSumOfDiff(J(:,:,i:i+2));
    bbox2 = getDetections(D2);
    feat2 = getFeatures(J(:,:,i+1),bbox2);
    
    if isempty(bbox2)
        disp('NO DETECTIONS')
        continue
    end
    
    M = getMatches(feat1,feat2);
    
    if isempty(M)
        disp('NO MATCHES')
        continue
    end
    
    subplot(2,1,1),drawnow,imshow(I(:,:,:,i)); title('Detections'); hold on
    for j = 1:size(bbox1,1)
        rectangle('Position',bbox1(j,:),'EdgeColor','r');
    end   
    
    subplot(2,1,2),showMatchedFeatures(I(:,:,:,i),I(:,:,:,i+1),[bbox1(M(:,1),1)+bbox1(M(:,1),3)/2 bbox1(M(:,1),2)+bbox1(M(:,1),4)/2], ...
        [bbox2(M(:,2),1)+bbox2(M(:,2),3)/2 bbox2(M(:,2),2)+bbox2(M(:,2),4)/2],'montage'); 
    title('Frame t                                         Frame t+1');
    
    D1 = D2;
    bbox1 = bbox2;
    feat1 = feat2;
end

