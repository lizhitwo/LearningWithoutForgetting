function net = vl_customnn_move(net, destination)
% VL_CUSTOMNN_MOVE  Move a custom CNN with probably Adam-style optimizer
% states between CPU and GPU
%    NET = VL_CUSTOMNN_MOVE(NET, 'gpu') moves the network
%    on the current GPU device.
%
%    NET = VL_CUSTOMNN_MOVE(NET, 'cpu') moves the network
%    on the CPU.
% 
% Authors: Zhizhong Li
% 
% See the COPYING file.

% Adapted from MatConvNet of VLFeat library. Their copyright info:

% Copyright (C) 2014 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

switch destination
  case 'gpu', moveop = @(x) gpuArray(x) ;
  case 'cpu', moveop = @(x) gather(x) ;
  otherwise, error('Unknown destination ''%s''.', destination) ;
end
for l=1:numel(net.layers)
  switch net.layers{l}.type
    case {'conv', 'bnorm', 'custom'}
      for f = {'filters', 'biases', 'filtersMomentum', 'biasesMomentum'}
        f = char(f) ;
        if isfield(net.layers{l}, f)
          net.layers{l}.(f) = moveop(net.layers{l}.(f)) ;
        end
      end
      for f = {'weights', 'momentum', 'momentum2', 'origweights'}
        f = char(f) ;
        if isfield(net.layers{l}, f)
          for j=1:numel(net.layers{l}.(f))
            net.layers{l}.(f){j} = moveop(net.layers{l}.(f){j}) ;
          end
        end
      end
    otherwise
      % nothing to do ?
  end
end
