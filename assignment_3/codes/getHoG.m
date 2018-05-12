function H = getHoG(I)

% I is a 2D matrix
% H if a vector of dimension 8 x 1

nbins = 16;
I = double(I);

Gx = I - [I(:,2:end) I(:,end)];
Gy = I - [I(2:end,:); I(end,:)];

ang = angle(Gx+Gy*1i)*180/pi; ang(:,end) = []; ang(end,:) = [];
bins = -180:360/nbins:180;
H = histc(ang(:), bins); H(end-1) = H(end-1) + H(end); H(end) = [];

H = H - mean(H); H = H / norm(H);
