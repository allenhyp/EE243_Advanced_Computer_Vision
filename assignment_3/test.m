I    = imread('./peppers_color.tif');    % Original: also test 2.jpg
I = imresize(I, [100, 100]);
SI   = 5;                  % Color similarity
SX   = 6;                  % Spatial similarity
r    = 1.5;                % Spatial threshold (less than r pixels apart)
sNcut = 0.21;              % The smallest Ncut value (threshold) to keep partitioning
sArea = 80;                % The smallest size of area (threshold) to be accepted as a segment
[Inc, Nnc]   = Nc(I,SI,SX,r,sNcut,sArea);   % Normalized Cut
subplot(236); imshow(Inc);  title(['NormalizedCut',' : ',num2str(Nnc)]); 

function [Inc Knc] = Nc(I,SI,SX,r,sNcut,sArea)

% Author: Naotoshi Seo <sonots(at)sonots.com>
% available online at: "http://note.sonots.com/SciSoftware/NcutImageSegmentation.html"

%% ncutImageSegment
[nRow, nCol,c] = size(I);                  % Changes
N = nRow * nCol;
V = reshape(I, N, c);                      % connect up-to-down way. Vertices of Graph
%% ncutComputeW                            
W = sparse(N,N);                           % Step 1. Compute weight matrix W, and D
F = reshape(I, N, 1, c);                   % col vector % Spatial Location
X = cat(3, repmat((1:nRow)', 1, nCol), repmat((1:nCol), nRow, 1));
X = reshape(X, N, 1, 2);                   % col vector

for ic=1:nCol                              % Future Work: Reduce computation to half. It can be done because W is symmetric mat
    for ir=1:nRow                          % matlab tricks for fast computation (Avoid 'for' loops as much as possible, instead use repmat.)
        
        % This range satisfies |X(i) - X(j)| <= r (block distance)
        jc = (ic - floor(r)) : (ic + floor(r)); % vector
        jr = ((ir - floor(r)) :(ir + floor(r)))';
        jc = jc(jc >= 1 & jc <= nCol);
        jr = jr(jr >= 1 & jr <= nRow);
        jN = length(jc) * length(jr);

        % index at vertex. V(i)
        i = ir + (ic - 1) * nRow;
        j = repmat(jr, 1, length(jc)) + repmat((jc -1) * nRow, length(jr), 1);
        j = reshape(j, length(jc) * length(jr), 1); % a col vector

        % spatial location distance (disimilarity)
        XJ = X(j, 1, :);
        XI = repmat(X(i, 1, :), length(j), 1);
        DX = XI - XJ;
        DX = sum(DX .* DX, 3); % squared euclid distance

        % |X(i) - X(j)| <= r (already satisfied if block distance measurement)
        constraint = find(sqrt(DX) <= r);
        j = j(constraint);
        DX = DX(constraint);

        % feature vector disimilarity
        FJ = F(j, 1, :);
        FI = repmat(F(i, 1, :), length(j), 1);
        DF = FI - FJ;
        DF = sum(DF .* DF, 3); % squared euclid distance ( DF = sum(abs(DF), 3); % block distance)
        W(i, j) = exp(-DF / (SI*SI)) .* exp(-DX / (SX*SX));
    end
end

%% ncutPartition
Seg = (1:N)';                             % Step 5. recursively repartition
id = 'ROOT';                              % the first segment has whole nodes. [1 2 3 ... N]'
% Compute D
N = length(W);
d = sum(W, 2);
D = spdiags(d, 0, N, N); % diagonal matrix
% Step 2 and 3. Solve generalized eigensystem (D -W)*S = S*D*U (12).
warning off; % let me stop warning
[U,S] = eigs(D-W, D, 2, 'sm');
% 2nd smallest (1st smallest has all same value elements, and useless)
U2 = U(:, 2);
% Bipartition the graph at point that Ncut is minimized.
t = mean(U2);
t = fminsearch('NcutValue', t, [], U2, W, D);
A = find(U2 > t);
B = find(U2 <= t);
% Step 4. Decide if the current partition should be divided
x = (U2 > t);
x = (2 * x) - 1;
d = diag(D);
k = sum(d(x > 0)) / sum(d);
b = k / (1 - k);
y = (1 + x) - b * (1 - x);
ncut = (y' * (D - W) * y) / ( y' * D * y );
%% itteration
if (length(A) < sArea || length(B) < sArea) || ncut > sNcut
    Seg{1}   = Seg;
    Id{1}   = id;   % for debugging
    Ncut{1} = ncut; % for duebugging
    return;
end
% Seg segments of A
[SegA IdA NcutA] = NcutPartition(Seg(A), W(A, A), sNcut, sArea, [id '-A']);
% Seg segments of B
[SegB IdB NcutB] = NcutPartition(Seg(B), W(B, B), sNcut, sArea, [id '-B']);
% concatenate cell arrays
Seg  = [SegA SegB];
Id   = [IdA IdB];
Ncut = [NcutA NcutB];
%% show
Inc  = zeros(size(I),'uint8');
for k=1:length(Seg)
 [r, c] = ind2sub(size(I),Seg{k});
 for i=1:length(r)
 Inc(r(i),c(i),1:3) = uint8(round(mean(V(Seg{k}, :))));
 end
end
Knc = length(Seg);

end

% needs:
% NcutPartition
% NcutValue

function ncut = NcutValue(t, U2, W, D)
% NcutValue - 2.1 Computing the Optimal Partition Ncut. eq (5)
%
% Synopsis
%  ncut = NcutValue(T, U2, D, W);
%
% Inputs ([]s are optional)
%  (scalar) t        splitting point (threshold)
%  (vector) U2       N x 1 vector representing the 2nd smallest
%                     eigenvector computed at step 2.
%  (matrix) W        N x N weight matrix
%  (matrix) D        N x N diagonal matrix
%
% Outputs ([]s are optional)
%  (scalar) ncut     The value calculated at the right term of eq (5).
%                    This is used to find minimum Ncut.
%
% Authors
%  Naotoshi Seo <sonots(at)sonots.com>
%
% License
%  The program is free to use for non-commercial academic purposes,
%  but for course works, you must understand what is going inside to use.
%  The program can be used, modified, or re-distributed for any purposes
%  if you or one of your group understand codes (the one must come to
%  court if court cases occur.) Please contact the authors if you are
%  interested in using the program without meeting the above conditions.

% Changes
%  10/01/2006  First Edition

%% NcutValue - Computing the Optimal Partition Ncut
x = (U2 > t);
x = (2 * x) - 1;
d = diag(D);
k = sum(d(x > 0)) / sum(d);
b = k / (1 - k);
y = (1 + x) - b * (1 - x);
ncut = (y' * (D - W) * y) / ( y' * D * y );
end

function [Seg Id Ncut] = NcutPartition(I, W, sNcut, sArea, id)
% NcutPartition - Partitioning
%
% Synopsis
%  [sub ids ncuts] = NcutPartition(I, W, sNcut, sArea, [id])
%
% Description
%  Partitioning. This function is called recursively.
%
% Inputs ([]s are optional)
%  (vector) I        N x 1 vector representing a segment to be partitioned.
%                    Each element has a node index of V (global segment).
%  (matrux) W        N x N matrix representing the computed similarity
%                    (weight) matrix.
%                    W(i,j) is similarity between node i and j.
%  (scalar) sNcut    The smallest Ncut value (threshold) to keep partitioning.
%  (scalar) sArea    The smallest size of area (threshold) to be accepted
%                    as a segment.
%  (string) [id]     A label of the segment (for debugg)
%
% Outputs ([]s are optional)
%  (cell)   Seg      A cell array of segments partitioned.
%                    Each cell is the each segment.
%  (cell)   Id       A cell array of strings representing labels of each segment.
%                    IDs are generated as children based on a parent id.
%  (cell)   Ncut     A cell array of scalars representing Ncut values
%                    of each segment.
%
% Requirements
%  NcutValue
%
% Authors
%  Naotoshi Seo <sonots(at)sonots.com>
%
% License
%  The program is free to use for non-commercial academic purposes,
%  but for course works, you must understand what is going inside to use.
%  The program can be used, modified, or re-distributed for any purposes
%  if you or one of your group understand codes (the one must come to
%  court if court cases occur.) Please contact the authors if you are
%  interested in using the program without meeting the above conditions.

% Changes
%  10/01/2006  First Edition
%% NcutPartition - Partitioning
N = length(W);
d = sum(W, 2);
D = spdiags(d, 0, N, N);    % diagonal matrix
% Step 2 and 3. Solve generalized eigensystem (D -W)*S = S*D*U (12).
warning off;                % let me stop warning
[U,S] = eigs(D-W, D, 2, 'sm');
% 2nd smallest (1st smallest has all same value elements, and useless)
U2 = U(:, 2);
% Bipartition the graph at point that Ncut is minimized.
t = mean(U2);
t = fminsearch('NcutValue', t, [], U2, W, D);
A = find(U2 > t);
B = find(U2 <= t);
% Step 4. Decide if the current partition should be divided
% if either of partition is too small, stop recursion.
% if Ncut is larger than threshold, stop recursion.
ncut = NcutValue(t, U2, W, D);
if (length(A) < sArea || length(B) < sArea) || ncut > sNcut
    Seg{1}   = I;
    Id{1}   = id;          % for debugging
    Ncut{1} = ncut;        % for duebuggin
    return;
end
% Seg segments of A
[SegA IdA NcutA] = NcutPartition(I(A), W(A, A), sNcut, sArea, [id '-A']);
% Seg segments of B
[SegB IdB NcutB] = NcutPartition(I(B), W(B, B), sNcut, sArea, [id '-B']);
% concatenate cell arrays
Seg   = [SegA SegB];
Id   = [IdA IdB];
Ncut = [NcutA NcutB];
end