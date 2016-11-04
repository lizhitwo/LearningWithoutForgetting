function [layers] = constructNewLayer(name, w_size, varargin)
% CONSTRUCTNEWLAYER   get MatConvNet layers from conv layer specs.
%   Input:
%     NAME (string) for naming the layers, W_SIZE (vector) for size of weights
%   Output:
%     LAYERS a cell of layers to be added to net.layers.
%   Options:
%     RELU: use 'relu' for adding relu afterwards and otherwise for not doing so
%     DROPOUT: positive for adding a dropout afterwards. 0 for no dropout.
% Note that the initialization is done elsewhere. Here the parameters are all 0
% 
% Authors: Zhizhong Li
% 
% See the COPYING file.

opts.relu = 'relu';
opts.dropout = 0.5;
opts = vl_argparse(opts, varargin);
assert(numel(w_size) == 4);

layers = cell(1,1);
layers{1} = struct('type', 'conv', 'name', ['fc' name], ...
    'pad', [0 0 0 0], 'stride', [1 1], ...
    'weights', []) ;
layers{1}.weights = cell(1,2); %      1           1        4096        4096
layers{1}.weights{1} = zeros(w_size, 'single');
layers{1}.weights{2} = zeros([1 w_size(4)], 'single');


switch opts.relu
    case 'relu'
        layers{end+1} = struct('type', 'relu', 'name', ['relu' name]) ;
    otherwise
end

if opts.dropout > 0
    layers{end+1} = struct('type', 'dropout', 'rate', opts.dropout) ;
end
