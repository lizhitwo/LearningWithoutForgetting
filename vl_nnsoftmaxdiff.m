function Y = vl_nnsoftmaxdiff(X,c,dzdy,opts)
% VL_NNSOFTMAXDIFF  combined softmax and distribution-difference loss. Supports L1/L2/cross-entropy
%    Y = VL_NNSOFTMAXDIFF(X, C) applies the softmax operator followed by
%    the difference between class distribution C and the data X. X has dimension H x W x D x N,
%    packing N arrays of W x H D-dimensional vectors.
%
%    Supporting three modes:
%     L1 - use a L1 loss for the distribution difference
%     L2 - use a L2 loss for the distribution difference
%     MI - use a Cross-entropy (a.k.a. Mutual Information) loss for the distribution difference.
%       A temperature can be specified. A temperature larger than 1 will make 
%       this a Knowledge Distillation loss.
%
%    C contains the target class distribution, which should have
%    D-dimensional distribution in the third dimension.
%    C can be an array with either D x N elements or with dimensions
%    H x W x D x N dimensions. In the fist case, a given class distribution is
%    applied at all spatial locations; in the second case, different
%    distributions can be specified for different locations.
%
%    DZDX = VL_NNSOFTMAXDIFF(X, C, DZDY) computes the derivative DZDX of the
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


% figure out which mode is being used
mode = opts.mode;
temperature = opts.temperature; 
switch mode
  case 'L1', assert(temperature == 1);
  case 'L2', assert(temperature == 1);
  case 'MI', assert(temperature > 0);
  otherwise, error('distribution loss mode not recognized');
end
if ~ isfield(opts, 'origstyle') 
    opts.origstyle = 'multiclass';
else
    if strcmp(opts.origstyle, 'multilabel')
      % right now for VOC style multilabel tasks, only MI is implemented.
      assert(strcmp(mode, 'MI'));
    end
end

%X = X + 1e-6 ;
sz = [size(X,1) size(X,2) size(X,3) size(X,4)] ;

if numel(c) == sz(3) * sz(4)
  % one label per image
  c = reshape(c, [1 1 sz(3) sz(4)]) ;
end
if size(c,1) == 1 & size(c,2) == 1
  c = repmat(c, [sz(1) sz(2)]) ;
end

normalize = sum(c,3);
assert(strcmp(opts.origstyle, 'multilabel') || sum(abs(normalize(:)-1)) < 1e-3);

% one label per spatial location
sz_ = [size(c,1) size(c,2) size(c,3) size(c,4)] ;
assert(isequal(sz_, [sz(1) sz(2) sz(3) sz(4)])) ;

% messy code that treats the target distributions with the temperature
if temperature ~= 1
  switch opts.origstyle
    case 'multiclass'
      c = c .^ (1/temperature);
      c = bsxfun(@rdivide, c, sum(c,3)) ;
    case 'multilabel'
      c_ = c .^ (1/temperature);
      c = c_ ./ (c_ + (1-c) .^ (1/temperature)) ;
    otherwise, assert(false, 'opts.origstyle not recognized');
  end
  
end



% compute softmaxloss
Xmax = max(X,[],3) ;
x_safe = bsxfun(@minus, X, Xmax)./temperature;
ex = exp(x_safe) ;
sumex = sum(ex,3);
assert(gather(all(sumex(:)>=0)));
softmax = bsxfun(@rdivide, ex, sumex) ;


% % debug
% if nargin <= 2 || isempty(dzdy)
%   fprintf('[%s, Temp=%.2f]', mode, temperature);
% end

% forward/backward pass for each mode and case
switch mode
  case 'L1'
    smdiff = softmax - c ;

    if nargin <= 2 || isempty(dzdy)
      Lp_diff = abs(smdiff);
      Y = sum(sum(sum(sum(Lp_diff,1),2),3),4) ;
    else
      dLpdiff_ddiff = sign(smdiff);
      dd_sm = dLpdiff_ddiff .* softmax;
      Y = dd_sm - bsxfun(@times, sum(dd_sm,3), softmax);
      Y = bsxfun(@times, Y, dzdy) ;
    end

  case 'L2'
    smdiff = softmax - c ;

    if nargin <= 2 || isempty(dzdy)
      Lp_diff = (smdiff.*smdiff)*.5;
      Y = sum(sum(sum(sum(Lp_diff,1),2),3),4) ;
    else
      dLpdiff_ddiff = smdiff;
      dd_sm = dLpdiff_ddiff .* softmax;
      Y = dd_sm - bsxfun(@times, sum(dd_sm,3), softmax);
      Y = bsxfun(@times, Y, dzdy) ;
    end

  case 'MI'
    switch opts.origstyle

      case 'multiclass' 
        % Cross-entropy for multiclass (categorical distribution)

        if nargin <= 2 || isempty(dzdy)
          % \mathcal{L} = - \sum_i { c_i * \log(softmax_i) }, but calculate in terms of x_safe is numerically better.
          Y = sum(sum(sum(log(sumex) - sum(c .* x_safe,3),1),2),4) ;
        else
          % its derivative happens to be this...
          Y = bsxfun(@times, softmax - c, dzdy) / temperature ;
        end

      case 'multilabel' 
        % Cross-entropy for multilabel (multiple separate binary distribution)
        
        X = X / temperature ;
        ndim_normalizer = 1; % use size(X,3) for averaging loss for different labels; 1 for summing loss for them.
        if nargin <= 2 || isempty(dzdy)
          % \mathcal{L} = - 1/ndim_normalizer * \sum_i { c_i * \log(softmax_i) + (1-c_i) * \log(1-softmax_i) }, 
          %             = 1/ndim_normalizer * \sum_i { \log(1+e^{X_i}) - x_i * c_i }
          % but calculate in a way numerically better.
          Y = log(1+exp(-abs(X))) + X.*(X > 0) - X.*c;
          Y = sum(Y(:)) / ndim_normalizer ;
        else
          Y = bsxfun(@times, 1./(1+exp(-X)) - c, dzdy) / temperature / ndim_normalizer ;
        end

      otherwise
        assert(false);
    end
  otherwise
    error('distribution loss mode not recognized');
end

%n = sz(1)*sz(2) ;
