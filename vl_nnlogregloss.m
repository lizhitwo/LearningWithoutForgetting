function Y = vl_nnlogregloss(X,c,dzdy)
% VL_NNLOGREGLOSS  CNN logistic loss for multiple **non-exclusive** label classification. e.g. that in VOC.
%    Y = VL_NNLOGREGLOSS(X, C) applies the logistic regression operator followed by
%    the logistic loss the data X. X has dimension H x W x D x N,
%    packing N arrays of W x H D-dimensional vectors.
%
%    C contains the class labels, which should be 0 and 1 labels for each class
%    with the same dimension as X. C can be an array with either D x N elements or with dimensions
%    H x W x D x N dimensions. In the fist case, a given class label is
%    applied at all spatial locations; in the second case, different
%    class labels can be specified for different locations.
%
%    DZDX = VL_NNLOGREGLOSS(X, C, DZDY) computes the derivative DZDX of the
%    function projected on the output derivative DZDY.
%    DZDX has the same dimension as X.

% Authors: Zhizhong Li
% 
% See the COPYING file.

% Adapted from MatConvNet of VLFeat library. Their copyright info:

% Copyright (C) 2014-15 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

%X = X + 1e-6 ;
sz = [size(X,1) size(X,2) size(X,3) size(X,4)] ;

if numel(c) == sz(3) * sz(4)
  % one label per image
  c = reshape(c, [1 1 sz(3) sz(4)]) ;
end
if size(c,1) == 1 & size(c,2) == 1
  c = repmat(c, [sz(1) sz(2)]) ;
end


% one label per spatial location
sz_ = [size(c,1) size(c,2) size(c,3) size(c,4)] ;
assert(isequal(sz_, [sz(1) sz(2) sz(3) sz(4)])) ;
assert(all(reshape((c==1) == c,[],1))) ; % all 0's and 1's



% compute logreg
gpuMode = isa(X, 'gpuArray'); % hack for my GPU issue
X = gather(X); c = gather(c);
p = vl_nnsigmoid(X);

% compute logit as: logit = logit_a * p_i + logit_b
logit_a = 2*c-1;
logit_b = 1-c;
logit = logit_a .* p + logit_b;
assert(all(logit(:)>=0));

if nargin <= 2
  logloss = -log(logit);
  Y = sum(sum(sum(sum(logloss,1),2),3),4) ;
  if gpuMode, Y = gpuArray(Y); end
else
  dzdp = -logit_a ./ logit;
  Y = vl_nnsigmoid(X,dzdp);
  Y = bsxfun(@times, Y, dzdy) ;
  if gpuMode, Y = gpuArray(Y); end
end
