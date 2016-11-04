function res = vl_graphnn(net, x, dzdy, res, varargin)
% VL_GRAPHNN  Evaluates a graph-structured CNN along the specified path.
%
% Paths:
%   VL_GRAPHNN supports parameter learning on a tree-structured network.
%   Different from vl_simplenn, this function evaluates / back propagates
%   along the root (input) to leaf (loss) path specified by the path number 
%   given by 'NN_PATH' optionally which refers to the index into net.paths{}.
%
%   The net will be treated as vl_simplenn if either net.paths or 'NN_PATH' 
%   is unspecified.
%
%   RES{} is only valid for layers that is along the specified path, except
%   for the output which is always in RES{end}. We use i_next to represent the 
%   next layer of #i according to net.paths{NN_PATH}. 
%
%   Also, support for siamese by passing a cell for input on both sides for
%   the sake of another project. Only conv, relu, etc. layers are supported.
%
%   Additionally a CUSTOM_SUDO layer is supported. Different from a CUSTOM 
%   layer, it can modify net and res (of other layers) directly.

% Authors: Zhizhong Li
% 
% See the COPYING file.

% Adapted from MatConvNet of VLFeat library. Their copyright info:

% Copyright (C) 2014 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

opts.res = [] ;
opts.conserveMemory = false ;
opts.sync = true ;
opts.disableDropout = false ;
opts.freezeDropout = false ;
opts.accumulate = false ; % accumulate gradient for the **weights**
opts.cudnn = true ;
opts.backPropDepth = +inf ;
opts.nn_path = 1; % the path (index into net.paths) to be used in this fwd/bkwd pass
opts.partial = []; % only fwd/bkwd pass on part of the layers. This argument specifies the range
opts.netAttachX = []; % if a function handle is provided, then this function is used to attach the input to net/res instead.

opts = vl_argparse(opts, varargin);

% figure out the root-to-leaf path and partial layers we are operating on
n = numel(net.layers) ;
if isfield(net, 'paths')
  path_layers = net.paths{opts.nn_path};
else
  path_layers = 1:n;
end
n_path_layers = numel(path_layers);
path_layers = [reshape(path_layers, 1, []), n+1]; % including the res{end} whichi is output 
if isempty(opts.partial)
  opts.partial = [1, n_path_layers];
else
  assert(1 <= opts.partial(1) && opts.partial(1) <= opts.partial(2) && opts.partial(2) <= n_path_layers);
end


% Do derivation?
if (nargin <= 2) || isempty(dzdy)
  doder = false ;
else
  doder = true ;
end

if opts.cudnn
  cudnn = {'CuDNN'} ;
else
  cudnn = {'NoCuDNN'} ;
end

if iscell(x), gpuMode = isa(x{1}, 'gpuArray');
else, gpuMode = isa(x, 'gpuArray') ;
end

% initialize RES
if nargin <= 3 || isempty(res)
  res = struct(...
    'x', cell(1,n+1), ...
    'dzdx', cell(1,n+1), ...
    'dzdw', cell(1,n+1), ...
    'aux', cell(1,n+1), ...
    'time', num2cell(zeros(1,n+1)), ...
    'backwardTime', num2cell(zeros(1,n+1))) ;
end

for iter=opts.partial(1):opts.partial(2)
  i = path_layers(iter);
  res(i).dzdx = [] ;
end
res(n+1).dzdx = [];

if gpuMode & opts.sync, wait(gpuDevice) ; end

if ~isa(opts.netAttachX, 'function_handle')
  res(1).x = x ;
else
  [net, res] = opts.netAttachX(net, res, x);
end

if gpuMode & opts.sync, wait(gpuDevice) ; end

% climbing through layers (forward)
for iter=opts.partial(1):opts.partial(2)
  i = path_layers(iter);
  i_next = path_layers(iter + 1);
  l = net.layers{i} ;
  res(i).time = tic ;
  switch l.type
    case 'conv'
      if isfield(l, 'weights')
        if iscell(res(i).x)
          % dealing with siamese
          nhalves = numel(res(i).x);
          res(i_next).x = cell(nhalves,1);
          for ihalf = 1:nhalves
            res(i_next).x{ihalf} = vl_nnconv(res(i).x{ihalf}, l.weights{1}, l.weights{2}, ...
                                   'pad', l.pad, 'stride', l.stride, ...
                                   cudnn{:}) ;
          end
        else % res(i).x not cell
          % regular non-siamese
          res(i_next).x = vl_nnconv(res(i).x, l.weights{1}, l.weights{2}, ...
                                 'pad', l.pad, 'stride', l.stride, ...
                                 cudnn{:}) ;
        end
      else % legacy 'filter/bias' naming
        res(i_next).x = vl_nnconv(res(i).x, l.filters, l.biases, ...
                               'pad', l.pad, 'stride', l.stride, ...
                               cudnn{:}) ;
      end
    case 'convt'
      if isfield(l, 'weights')
        res(i_next).x = vl_nnconvt(res(i).x, l.weights{1}, l.weights{2}, ...
                               'crop', l.crop, 'upsample', l.upsample, ...
                               cudnn{:}) ;
      else
        res(i_next).x = vl_nnconv(res(i).x, l.filters, l.biases, ...
                               'crop', l.pad, 'upsample', l.upsample, ...
                               cudnn{:}) ;
      end
    case 'pool'
      res(i_next).x = vl_nnpool(res(i).x, l.pool, ...
                             'pad', l.pad, 'stride', l.stride, ...
                             'method', l.method, ...
                             cudnn{:}) ;
    case 'normalize'
      res(i_next).x = vl_nnnormalize(res(i).x, l.param) ;
    case 'softmax'
      res(i_next).x = vl_nnsoftmax(res(i).x) ;
    case 'loss'
      res(i_next).x = vl_nnloss(res(i).x, l.class) ;
    case 'softmaxloss'
      res(i_next).x = vl_nnsoftmaxloss(res(i).x, l.class) ;
    case 'relu'
      if iscell(res(i).x)
        % dealing with siamese
        nhalves = numel(res(i).x);
        res(i_next).x = cell(nhalves,1);
        for ihalf = 1:nhalves
          res(i_next).x{ihalf} = vl_nnrelu(res(i).x{ihalf}) ;
        end
      else % res(i).x not cell
        % regular non-siamese
        res(i_next).x = vl_nnrelu(res(i).x) ;
      end
    case 'sigmoid'
      res(i_next).x = vl_nnsigmoid(res(i).x) ;
    case 'noffset'
      res(i_next).x = vl_nnnoffset(res(i).x, l.param) ;
    case 'spnorm'
      res(i_next).x = vl_nnspnorm(res(i).x, l.param) ;
    case 'dropout'
      if opts.disableDropout
        res(i_next).x = res(i).x ;
      elseif opts.freezeDropout
        if iscell(res(i).x)
          % dealing with siamese
          nhalves = numel(res(i).x);
          res(i_next).x = cell(nhalves,1);
          for ihalf = 1:nhalves
            [res(i_next).x{ihalf}, res(i_next).aux{ihalf}] = vl_nndropout(res(i).x{ihalf}, 'rate', l.rate, 'mask', res(i_next).aux{ihalf}) ;
          end
        else % res(i).x not cell
          % regular non-siamese
          [res(i_next).x, res(i_next).aux] = vl_nndropout(res(i).x, 'rate', l.rate, 'mask', res(i_next).aux) ;
        end
      else % do not freeze/disable dropout
        if iscell(res(i).x)
          % dealing with siamese
          nhalves = numel(res(i).x);
          res(i_next).x = cell(nhalves,1);
          for ihalf = 1:nhalves
            if ihalf==1, [res(i_next).x{ihalf}, res(i_next).aux] = vl_nndropout(res(i).x{ihalf}, 'rate', l.rate) ; % dropout determined on 1st siamese branch
            else [res(i_next).x{ihalf}, res(i_next).aux] = vl_nndropout(res(i).x{ihalf}, 'rate', l.rate, 'mask', res(i_next).aux) ;
            end
          end
        else % res(i).x not cell
          % regular non-siamese
          [res(i_next).x, res(i_next).aux] = vl_nndropout(res(i).x, 'rate', l.rate) ;
        end
        
      end
    case 'bnorm'
      if isfield(l, 'weights')
        res(i_next).x = vl_nnbnorm(res(i).x, l.weights{1}, l.weights{2}) ;
      else
        res(i_next).x = vl_nnbnorm(res(i).x, l.filters, l.biases) ;
      end
    case 'pdist'
      res(i_next) = vl_nnpdist(res(i).x, l.p, 'noRoot', l.noRoot, 'epsilon', l.epsilon) ;
    case 'custom'
      res(i_next) = l.forward(l, res(i), res(i_next)) ;
    case 'custom_sudo'
      [net, res] = l.forward(net, res, i, i_next) ;
    otherwise
      error('Unknown layer type %s', l.type) ;
  end
  % optionally forget intermediate results
  forget = opts.conserveMemory ;
  forget = forget & (~doder || strcmp(l.type, 'relu')) ;
  forget = forget & ~(strcmp(l.type, 'loss') || strcmp(l.type, 'softmaxloss')) ;
  forget = forget & (~isfield(l, 'rememberOutput') || ~l.rememberOutput) ;
  forget = forget & iter ~= n_path_layers;
  if forget
    res(i).x = [] ;
  end
  if gpuMode & opts.sync
    % This should make things slower, but on MATLAB 2014a it is necessary
    % for any decent performance.
    wait(gpuDevice) ;
  end
  res(i).time = toc(res(i).time) ;
end


% walking down the layers (bkwd pass)
if doder
  res(n+1).dzdx = dzdy ;
  for iter=opts.partial(2):-1:max(opts.partial(1), n_path_layers-opts.backPropDepth+1)
    i = path_layers(iter);
    i_next = path_layers(iter+1);
    l = net.layers{i} ;
    res(i).backwardTime = tic ;
    orig_dzdx = res(i).dzdx; % allowing for accumulating gradients for **dzdx** which can be used to merge gradients from multiple sources
    switch l.type
      case 'conv'
        if ~opts.accumulate && ~(isfield(l, 'accumulate_once') && l.accumulate_once)
          if isfield(l, 'weights')
            if iscell(res(i).x)
              % dealing with siamese
              nhalves = numel(res(i).x);
              res(i).dzdx = cell(nhalves,1); res(i).dzdw = [];
              for ihalf = 1:nhalves
                [res(i).dzdx{ihalf}, dzdw1, dzdw2] = ...
                    vl_nnconv(res(i).x{ihalf}, l.weights{1}, l.weights{2}, ...
                              res(i_next).dzdx{ihalf}, ...
                              'pad', l.pad, 'stride', l.stride, ...
                              cudnn{:}) ;
                % accumulate gradients for siamese!
                if isempty(res(i).dzdw), res(i).dzdw = {dzdw1, dzdw2}; 
                else res(i).dzdw{1} = res(i).dzdw{1}+dzdw1; res(i).dzdw{2} = res(i).dzdw{2}+dzdw2; end;
                clear dzdw1 dzdw2;
              end
            else % res(i).x not cell
              % regular non-siamese
              [res(i).dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = ...
                  vl_nnconv(res(i).x, l.weights{1}, l.weights{2}, ...
                            res(i_next).dzdx, ...
                            'pad', l.pad, 'stride', l.stride, ...
                            cudnn{:}) ;
            end
          else
            % Legacy code: will go
            [res(i).dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = ...
                vl_nnconv(res(i).x, l.filters, l.biases, ...
                          res(i_next).dzdx, ...
                          'pad', l.pad, 'stride', l.stride, ...
                          cudnn{:}) ;
          end
        else % opts.accumulate
          net.layers{i}.accumulate_once = false;
          dzdw = cell(1,2) ;
          if isfield(l, 'weights')
            if iscell(res(i).x)
              % dealing with siamese
              nhalves = numel(res(i).x);
              res(i).dzdx = cell(nhalves,1); dzdw = [];
              for ihalf = 1:nhalves
                [res(i).dzdx{ihalf}, dzdw1, dzdw2] = ...
                    vl_nnconv(res(i).x{ihalf}, l.weights{1}, l.weights{2}, ...
                              res(i_next).dzdx{ihalf}, ...
                              'pad', l.pad, 'stride', l.stride, ...
                              cudnn{:}) ;
                % accumulate gradients for siamese!
                if isempty(dzdw), dzdw = {dzdw1, dzdw2}; 
                else dzdw{1} = dzdw{1}+dzdw1; dzdw{2} = dzdw{2}+dzdw2; end;
                clear dzdw1 dzdw2;
              end
            else % res(i).x not cell
              % regular non-siamese
              [res(i).dzdx, dzdw{1}, dzdw{2}] = ...
                  vl_nnconv(res(i).x, l.weights{1}, l.weights{2}, ...
                            res(i_next).dzdx, ...
                            'pad', l.pad, 'stride', l.stride, ...
                            cudnn{:}) ;
            end
          else
            % Legacy code: will go
            [res(i).dzdx, dzdw{1}, dzdw{2}] = ...
                vl_nnconv(res(i).x, l.filters, l.biases, ...
                          res(i_next).dzdx, ...
                          'pad', l.pad, 'stride', l.stride, ...
                          cudnn{:}) ;
          end
          % because opts.accumulate
          if isempty(res(i).dzdw), res(i).dzdw = dzdw;
          else
            for j=1:2
              res(i).dzdw{j} = res(i).dzdw{j} + dzdw{j} ;
            end
          end
          clear dzdw ;
        end

      case 'convt'
        if ~opts.accumulate
          if isfield(l, 'weights')
            [res(i).dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = ...
                vl_nnconvt(res(i).x, l.weights{1}, l.weights{2}, ...
                          res(i_next).dzdx, ...
                          'crop', l.crop, 'upsample', l.upsample, ...
                          cudnn{:}) ;
          else
            % Legacy code: will go
            [res(i).dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = ...
                vl_nnconvt(res(i).x, l.filters, l.biases, ...
                         res(i_next).dzdx, ...
                          'crop', l.crop, 'upsample', l.upsample, ...
                          cudnn{:}) ;
          end
        else % opts.accumulate
          dzdw = cell(1,2) ;
          if isfield(l, 'weights')
            [res(i).dzdx, dzdw{1}, dzdw{2}] = ...
                vl_nnconvt(res(i).x, l.weights{1}, l.weights{2}, ...
                          res(i_next).dzdx, ...
                          'crop', l.crop, 'upsample', l.upsample, ...
                          cudnn{:}) ;
          else
            % Legacy code: will go
            [res(i).dzdx, dzdw{1}, dzdw{2}] = ...
                vl_nnconvt(res(i).x, l.filters, l.biases, ...
                          res(i_next).dzdx, ...
                          'crop', l.crop, 'upsample', l.upsample, ...
                          cudnn{:}) ;
          end
          if isempty(res(i).dzdw), res(i).dzdw = dzdw;
          else
            for j=1:2
              res(i).dzdw{j} = res(i).dzdw{j} + dzdw{j} ;
            end
          end
          clear dzdw ;
        end
       
      case 'pool'
        res(i).dzdx = vl_nnpool(res(i).x, l.pool, res(i_next).dzdx, ...
                                'pad', l.pad, 'stride', l.stride, ...
                                'method', l.method, ...
                                cudnn{:}) ;
      case 'normalize'
        res(i).dzdx = vl_nnnormalize(res(i).x, l.param, res(i_next).dzdx) ;
      case 'softmax'
        res(i).dzdx = vl_nnsoftmax(res(i).x, res(i_next).dzdx) ;
      case 'loss'
        res(i).dzdx = vl_nnloss(res(i).x, l.class, res(i_next).dzdx) ;
      case 'softmaxloss'
        res(i).dzdx = vl_nnsoftmaxloss(res(i).x, l.class, res(i_next).dzdx) ;
      case 'relu'
        if iscell(res(i_next).dzdx)
          % dealing with siamese
          nhalves = numel(res(i_next).dzdx);
          res(i).dzdx = cell(nhalves,1);
          if ~isempty(res(i).x)
            for ihalf = 1:nhalves
              res(i).dzdx{ihalf} = vl_nnrelu(res(i).x{ihalf}, res(i_next).dzdx{ihalf}) ;
            end
          else
            % if res(i).x is empty, it has been optimized away, so we use this
            % hack (which works only for ReLU):
            for ihalf = 1:nhalves
              res(i).dzdx{ihalf} = vl_nnrelu(res(i_next).x{ihalf}, res(i_next).dzdx{ihalf}) ;
            end
          end
        else % res(i_next).dzdx not cell
          % regular non-siamese
          if ~isempty(res(i).x)
            res(i).dzdx = vl_nnrelu(res(i).x, res(i_next).dzdx) ;
          else
            % if res(i).x is empty, it has been optimized away, so we use this
            % hack (which works only for ReLU):
            res(i).dzdx = vl_nnrelu(res(i_next).x, res(i_next).dzdx) ;
          end
        end
      case 'sigmoid'
        res(i).dzdx = vl_nnsigmoid(res(i).x, res(i_next).dzdx) ;
      case 'noffset'
        res(i).dzdx = vl_nnnoffset(res(i).x, l.param, res(i_next).dzdx) ;
      case 'spnorm'
        res(i).dzdx = vl_nnspnorm(res(i).x, l.param, res(i_next).dzdx) ;
      case 'dropout'
        if opts.disableDropout
          res(i).dzdx = res(i_next).dzdx ;
        else % not disabled
          if iscell(res(i_next).dzdx)
            % dealing with siamese
            nhalves = numel(res(i_next).dzdx);
            res(i).dzdx = cell(nhalves,1);
            for ihalf = 1:nhalves
              res(i).dzdx{ihalf} = vl_nndropout(res(i).x{ihalf}, res(i_next).dzdx{ihalf}, ...
                                         'mask', res(i_next).aux) ;
            end
          else % res(i_next).dzdx not cell
            % regular non-siamese
            res(i).dzdx = vl_nndropout(res(i).x, res(i_next).dzdx, ...
                                       'mask', res(i_next).aux) ;
          end
        end
      case 'bnorm'
        if ~opts.accumulate
          if isfield(l, 'weights')
            [res(i).dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = ...
                vl_nnbnorm(res(i).x, l.weights{1}, l.weights{2}, ...
                           res(i_next).dzdx) ;
          else
            [res(i).dzdx, res(i).dzdw{1}, res(i).dzdw{2}] = ...
                vl_nnbnorm(res(i).x, l.filters, l.biases, ...
                           res(i_next).dzdx) ;
          end
        else
          dzdw = cell(1,2) ;
          if isfield(l, 'weights')
            [res(i).dzdx, dzdw{1}, dzdw{2}] = ...
                vl_nnbnorm(res(i).x, l.weights{1}, l.weights{2}, ...
                           res(i_next).dzdx) ;
          else
            [res(i).dzdx, dzdw{1}, dzdw{2}] = ...
                vl_nnbnorm(res(i).x, l.filters, l.biases, ...
                           res(i_next).dzdx) ;
          end
          if isempty(res(i).dzdw), res(i).dzdw = dzdw;
          else
            for j=1:2
              res(i).dzdw{j} = res(i).dzdw{j} + dzdw{j} ;
            end
          end
          clear dzdw ;
        end
      case 'pdist'
        res(i).dzdx = vl_nnpdist(res(i).x, l.p, res(i_next).dzdx, ...
                                 'noRoot', l.noRoot, 'epsilon', l.epsilon) ;
      case 'custom'
        res(i) = l.backward(l, res(i), res(i_next)) ;
      case 'custom_sudo'
        [net, res] = l.backward(net, res, i, i_next) ;
    end

    % allowing for accumulating gradients for **dzdx** which can be used to merge gradients from multiple sources
    if ~isempty(orig_dzdx)
      if iscell(orig_dzdx)
        for ihalf = numel(orig_dzdx)
          res(i).dzdx{ihalf} = res(i).dzdx{ihalf} + orig_dzdx{ihalf};
        end
      else
        res(i).dzdx = res(i).dzdx + orig_dzdx;
      end
    end

    if opts.conserveMemory
      res(i_next).dzdx = [] ;
    end
    if gpuMode & opts.sync
      wait(gpuDevice) ;
    end
    res(i).backwardTime = toc(res(i).backwardTime) ;
  end
end
