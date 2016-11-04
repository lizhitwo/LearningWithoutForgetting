function [ dimlabel ] = dimensionalizeLabel( label, n_classes )
% DIMENSIONALIZELABEL   convert from enumerated labels to one-hot vector labels
%   Input:
%     LABEL (vector) e.g. [1 2 2 1 2]
%     N_CLASSES the number of classes, e.g. 3
%   Output: 
%     DIMLABEL matrix of one-hot vectors. e.g. [1 0 0 1 0; 0 1 1 0 1; 0 0 0 0 0]
% 
% Authors: Zhizhong Li
% 
% See the COPYING file.

assert(max(size(label)) == numel(label));
label = label(:);
n_samples = numel(label);
if nargin >= 2
    maxlbl = n_classes;
    assert(max(label) <= maxlbl);
else
    maxlbl = max(label);
end
minlbl = min(label);
assert(minlbl >= 1);
dimlabel = zeros(maxlbl, n_samples);
dimlabel(sub2ind(size(dimlabel), label, (1:n_samples)')) = 1;