function [ check ] = testLayer(  )
% TESTLAYER   Check numerically whether a layer's backprop is implemented correctly
% 
% Authors: Zhizhong Li
% 
% See the COPYING file.

fn_str = 'vl_nnsoftmaxdiff'; % name of the layer to test;

xsize = [1 1 30 20];
csize = [1 1 30 20];
xscale = 2;
nruns = 1000;
fnopts = [];
fnopts = struct('mode', 'MI', 'temperature', 2, 'origstyle', 'multilabel');

opts.classtype = 'distrib'; % distrib
opts.diffstep = 0.0001;
opts.tolerance = 1e-6;

fn = eval(['@' fn_str]);

for irun = 1:nruns
    % init rand for x,c,w, whatnots
    x0 = xscale*randn(xsize);
    switch opts.classtype
        case 'distrib'
            c = abs(randn(csize));
            c = bsxfun(@rdivide, c, sum(c,3));
        case '0-1'
            c = randi([0 1], csize);
    end
    if irun==0
        % c(1,1,:,:) = [.25, .5, .25; .25, .5, .25]';
        % x0(1,1,:,:) = [0, log(2), 0; 0, log(2), 0]';
        x0 = x0*0;
        x0(:,:,1,:) = c(:,:,2,:);
    end
    if isempty(fnopts)
        y0 = fn(x0,c);
    else
        y0 = fn(x0,c,[],fnopts);
    end
    dzdy = 1*randn(size(y0));
    if isempty(fnopts)
        dzdx0 = fn(x0,c,dzdy);
    else
        dzdx0 = fn(x0,c,dzdy,fnopts);
    end
    dzdx_num = zeros(size(dzdx0));
    
    % for x
    for i=1:numel(x0)
        x = x0;
        x(i) = x0(i) + opts.diffstep;
        if isempty(fnopts)
            y_p = fn(x,c);
        else
            y_p = fn(x,c,[],fnopts);
        end
        x(i) = x0(i) - opts.diffstep;
        if isempty(fnopts)
            y_n = fn(x,c);
        else
            y_n = fn(x,c,[],fnopts);
        end
        dydxi = (y_p-y_n) / (2*opts.diffstep);
        dzdxi = dzdy .* dydxi;
        dzdxi = sum(dzdxi(:));
        dzdx_num(i) = dzdxi;
    end
    
    % differentiate them
    diff = abs(dzdx0-dzdx_num);
    totdiff = max(diff(:));
    disp(totdiff);
    
    % bug found
    if totdiff >= opts.tolerance
        % is it because of singularity?
        singularity = true;
        
        % singularity for each function
        switch fn_str
            case 'vl_nnsoftmaxdiff'
                inds = find(diff>opts.tolerance);
                for i=inds(:)'
                    [i,j] = ind2sub([size(diff,3),size(diff,4)],i);
                    p = sum(exp(x(1,1,:,j)-x(1,1,i,j)),3);
                    if abs(1/p - c(1,1,i,j))>opts.tolerance
                        singularity = false;
                    end
                end
                singularity = false;
            case 'vl_nnlogregloss'
                singularity = false;
            otherwise
                singularity = false;
        end
        
        if ~singularity
            keyboard;
        end
        assert(singularity);
    end
end

check = 1;


% -------------------------------------------------------------------------
function Y = vl_nnrankofL2(X,c,dzdy)
% VL_NNLOGREGLOSS  CNN logistic loss for multi(-independent-)label classification

% Zhizhong Li 2015
% Mimicking MatConvNet of VLFeat library. Their copyright info:

% Copyright (C) 2014-15 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

lambda_sim = 1/(0.25 ^ 2);
gamma = 10;
%X = X + 1e-6 ;
sz = [size(X,1) size(X,2) size(X,3) size(X,4)] ;
sizehalf = sz(3)/2;
assert(sizehalf==floor(sizehalf));
assert( numel(c) == 2 * sz(4) );
c = reshape(c, [1 1 2 sz(4)]) ;

c_class = sign(c(:,:,1,:)-0.5);
c_meshsim = c(:,:,2,:);

% compute diff & Xsim & sqloss of Xsim
X1 = X(:,:,1:sizehalf,:);
X2 = X(:,:,sizehalf+1:end,:);
diffX = X1 - X2;

% similarity
% Xsim = sqrt(sum(diffX .* diffX, 3)); % L2
sumX1squ = sum(X1.*X1, 3);
sumX2squ = sum(X2.*X2, 3);
sumX1X2 = sum(X1.*X2, 3);
sqrtsumX1_sqrtsumX2 = sqrt(sumX1squ .* sumX2squ);;
Xsim = sumX1X2 ./ sqrtsumX1_sqrtsumX2;

c_meshsim_roll = cat(4, c_meshsim(:,:,:,2:end), c_meshsim(:,:,:,1));
Xsim_roll = cat(4, Xsim(:,:,:,2:end), Xsim(:,:,:,1));

gt_gtnext = sign(c_meshsim - c_meshsim_roll);
gammaDelta = gamma * gt_gtnext .* (Xsim - Xsim_roll);

% signDelta = sign(gammaDelta);

if nargin <= 2
  safelogloss = log(1+exp(-abs(gammaDelta)));
  safelogloss = safelogloss - (gammaDelta .* (gammaDelta < 0));
  Y = lambda_sim * sum(sum(sum(sum(safelogloss,1),2),3),4) ;
else
  dzdDelta = -gamma./(1+exp(gammaDelta));

  dzdXsim_part1 = dzdDelta .* gt_gtnext;
  dzdXsim = dzdXsim_part1 - cat(4, dzdXsim_part1(:,:,:,end), dzdXsim_part1(:,:,:,1:end-1));
  % for L2
  % Y = bsxfun(@times, dzdXsim , bsxfun(@rdivide, diffX, Xsim));
  % Y = bsxfun(@times, cat(3,Y, -Y), lambda_sim * dzdy) ;
  % for cos
  Y = cat(3,X2 - bsxfun(@times, sumX1X2./sumX1squ, X1), X1 - bsxfun(@times, sumX1X2./sumX2squ, X2));
  Y = bsxfun(@times, Y, dzdXsim);
  Y = bsxfun(@rdivide, Y, sqrtsumX1_sqrtsumX2) ;
  Y = bsxfun(@times, Y, lambda_sim * dzdy) ;
end


