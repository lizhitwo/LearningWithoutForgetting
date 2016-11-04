function [net] = forkAlexNet(net, new_layers, varargin)
% EXPANDALEXNET   basically, construct a graphnn from the AlexNet or VGG-16 by expanding the network dimensions.
%   Input:
%     NEW_LAYERS (scalar) the number of task-specific layers. 
%     !!! Not to be confused with the NEW_LAYERS argument, which is the number of added layers on top of that!
%   Options:
%     See code comments
% 
% Authors: Zhizhong Li
% 
% See the COPYING file.

opts.mode = 'multiclass'; % type of last layer of the new path
opts.newtaskdim = 20; % # output of last layer
opts.init_scale = 0.01; % scale for randomly initializing new neurons
opts.weightInitMethod = 'gaussian'; % when randomly initializing, gaussian (fixed scale) or glorot (xavier, induce scale based on input/output of the layer)
opts.init_bias = 0; % what to init the biases as
opts.redo_layers = 2; % the number of existing layers to be re-initialized (old weights thrown away)
opts.new_layers = 0; % the number of **additional** layers to be added in the new-task-specific layers
opts.copy_source = 1; % which net.paths{} path should we copy from
opts.orig_loss = 'for_training'; % for_training: softmaxloss, for_eval: softmax, for_keep: custom, else: don't touch
opts.keep_response_loss = 'MI'; % MI for mutual information, L1 for L1-norm. only works when orig_loss is 'for_keep'.
opts.distillation_temp = 2; % only works when keepresponse_method is 'MI'.
opts.origfc_adddropout = true;
opts = vl_argparse(opts, varargin);

networkver = isfield(net, 'paths'); % 1 for vl_graphnn, 0 for vl_simplenn
% modify net to add dropouts... because the given networks do not have them
if opts.origfc_adddropout
    assert(networkver == 0, 'graphnn should already have dropout');
    n_layers_orig = numel(net.layers);
    assert(n_layers_orig == 21 || n_layers_orig == 37);
    net.layers{end+1} = struct('type', 'dropout', 'rate', 0.5) ;
    net.layers{end+1} = struct('type', 'dropout', 'rate', 0.5) ;
    % net.layers = net.layers([ 1:15, 16,17,22, 18,19,23, 20,21 ]); % for AlexNet
    net.layers = net.layers([ 1:(n_layers_orig-6), n_layers_orig + [-5,-4,1, -3,-2,2, -1,0] ]); % AlexNet and VGG
end

if networkver == 0, net.paths = {1:numel(net.layers)}; end
n_paths_orig = numel(net.paths);
n_layers_orig = numel(net.paths{opts.copy_source});
assert( isnumeric(new_layers) && numel(new_layers)==1 );
assert(new_layers > 0, 'Must copy SOME layer; otherwise, what are you doing');
assert(opts.redo_layers > 0, 'Assuming the last weight should be replaced.');

% determine num of copied-over new layers
n_layers_conv = 0;
n_layers_copied = 0;
n_layers_redo = 0;
for i=1:n_layers_orig
    if strcmp(net.layers{net.paths{opts.copy_source}(end-i+1)}.type, 'conv')
        n_layers_conv = n_layers_conv + 1;
    end
    if n_layers_conv == new_layers && ~n_layers_copied, n_layers_copied = i; end
    if n_layers_conv == opts.redo_layers && ~n_layers_redo, n_layers_redo = i; end
    if n_layers_copied && n_layers_redo
        break;
    end
end

assert(n_layers_copied > 0, 'Cannot branch and copy more layers than there are');
assert(n_layers_redo > 0, 'Cannot re-initialize more layers than there are');
assert(n_layers_redo <= n_layers_copied, 'Cannot re-initialize layers that are not branched');

% deal with "for_keep"'s adding a keep response loss layer.
% [ HACK ] Now, only adding the LAST task's keep response loss layer is supported. 
% "Last task" means the original task for the first new task, and the last new task for all following new tasks.
if strcmp(opts.orig_loss, 'for_keep')
    % task "n+1" [ HACK ] GUESSING WHAT THE LAST LOSS SHOULD BE without specifying it using a input! 
    % [ HACK ] Now, we guess it's multiclass for 1st and the others are the same as new task!
    net.layers{end+1}.type = 'custom';
    net.layers{end}.name = 'keepold';
    if n_paths_orig > 1 % keeping response for some layer like the new task form!
        switch opts.mode
            case {'multiclass', 'multiclasshinge'}
                layeropts.origstyle = 'multiclass';
            case 'multilabel'
                layeropts.origstyle = 'multilabel';
        end
    else % keeping response for some layer like the old task form!
        layeropts.origstyle = 'multiclass';
    end
    layeropts.temperature = opts.distillation_temp;
    layeropts.mode = opts.keep_response_loss;
    net.layers{end}.forward = getFwdHandle( @vl_nnsoftmaxdiff, layeropts );
    net.layers{end}.backward = getBkwdHandle( @vl_nnsoftmaxdiff, layeropts );
    % insert before new task
    net.paths = { net.paths{:}, [ net.paths{end}(1:end-1), numel(net.layers) ] };
end

% copy layers
n_layers_orig_allpath = numel(net.layers);
net.layers(end+1:end+n_layers_copied) = net.layers(net.paths{opts.copy_source}(end-n_layers_copied+1:end));
net.layers{end}.type = 'softmaxloss';
net.layers{end}.name = 'newtaskloss';
if n_paths_orig > 1, net.layers{end}.name = sprintf('newtask%dloss', n_paths_orig+1); end


% brand new layers to add to the new-task-specific layers
newlayers = {};
for i=1:opts.new_layers
    newlayers = [newlayers, constructNewLayer(sprintf('+%d', i), [1 1 4096 4096])]; % only construct them; the weights are initialized later
end
n_layers_new = n_layers_copied + numel(newlayers);
n_layers_redo = n_layers_redo + numel(newlayers);
assert(n_layers_copied >= 2, 'Assuming copying over at least a loss layer and its corresponding weight layer');
net.layers = [ net.layers(1:end-2), newlayers, net.layers(end-1:end) ];

% update paths to reflect newly added path and newly added layers
net.paths = { net.paths{:}, [net.paths{opts.copy_source}(1:(n_layers_orig-n_layers_copied)), n_layers_orig_allpath+(1:n_layers_new)] };


% change new task loss and # classes
new_fc_last = n_layers_orig_allpath+n_layers_new-1; assert(new_fc_last == net.paths{end}(end-1));
switch opts.mode
    case {'multiclass', 'multiclasshinge'}
        for i = 1:numel(net.layers{new_fc_last}.weights)
            weight_size = size(net.layers{new_fc_last}.weights{i});
            weight_size(end) = opts.newtaskdim;
            net.layers{new_fc_last}.weights{i} = zeros(weight_size, 'single');
        end
        % type defaults to 'softmaxloss', according to what we did when copying layers
        if strcmp(opts.mode, 'multiclasshinge')
            net.layers{end}.type = 'custom';
            net.layers{end}.name = 'multiclasshinge';
            if n_paths_orig > 1, net.layers{end}.name = sprintf('multiclasshinge%d', n_paths_orig+1); end
            layeropts.loss = 'mshinge';
            net.layers{end}.forward = getFwdHandle( @vl_nnloss_future, layeropts );
            net.layers{end}.backward = getBkwdHandle( @vl_nnloss_future, layeropts );
        end
    case 'multilabel'
        for i = 1:numel(net.layers{new_fc_last}.weights)
            weight_size = size(net.layers{new_fc_last}.weights{i});
            weight_size(end) = opts.newtaskdim;
            net.layers{new_fc_last}.weights{i} = zeros(weight_size, 'single');
        end
        net.layers{end}.type = 'custom';
        net.layers{end}.name = 'multilogreg';
        if n_paths_orig > 1, net.layers{end}.name = sprintf('multilogreg%d', n_paths_orig+1); end
        net.layers{end}.forward = getFwdHandle( @vl_nnlogregloss );
        net.layers{end}.backward = getBkwdHandle( @vl_nnlogregloss );
    otherwise
end
% rand init (not sure what to do about other fields e.g. momentum)
for i_layers = net.paths{end}(end-n_layers_redo+1:end)
    if ~isfield(net.layers{i_layers}, 'weights'), continue; end
    switch opts.weightInitMethod
        case 'gaussian'
            sc = opts.init_scale ;
        case 'glorot'
            if ~isa(net.layers{i_layers}.weights, 'cell')
                sz = size(net.layers{i_layers}.weights);
            else
                sz = size(net.layers{i_layers}.weights{1});
            end
            sc = sqrt(1/(sz(1)*sz(2)*(sz(3)+sz(4))/opts.init_scale)) ;
        otherwise
            throw(MException('forkAlexNet:weightInitMethod','Unrecognized weightInitMethod value'));
    end
    if ~isa(net.layers{i_layers}.weights, 'cell')
        % only deals with "weight only"
        % using 'glorot'
        net.layers{i_layers}.weights = sc * ...
            randn(size(net.layers{i_layers}.weights), 'single');
    else
        % only deals with "weight+bias combination"
        assert(numel(net.layers{i_layers}.weights) == 2);
        % using 'glorot'
        net.layers{i_layers}.weights{1} = single(sc) * ...
            randn(size(net.layers{i_layers}.weights{1}), 'single');
        net.layers{i_layers}.weights{2} = single(opts.init_bias) + ...
            zeros(size(net.layers{i_layers}.weights{2}), 'single');
    end
end

% deal with different loss
switch opts.orig_loss
    case 'for_training'
        net.layers{n_layers_orig}.type = 'softmaxloss';
        net.layers{n_layers_orig}.name = 'loss';
    case 'for_eval'
        net.layers{n_layers_orig}.type = 'softmax';
        net.layers{n_layers_orig}.name = 'prob';
    case 'for_keep'
        net.layers{n_layers_orig}.type = 'softmaxloss';
        net.layers{n_layers_orig}.name = 'origloss';
end


% -------------------------------------------------------------------------
function [ fo ] = getFwdHandle( fi, opts )
if nargin < 2, opts = []; end
fo = @(layer, resi, resi_next) Forward(layer, resi, resi_next, fi, opts);

% -------------------------------------------------------------------------
function [ resi_next ] = Forward(layer, resi, resi_next, fi, opts)
if isempty(opts)
    resi_next.x = fi(resi.x, layer.class);
else
    resi_next.x = fi(resi.x, layer.class, [], opts);
end

% -------------------------------------------------------------------------
function [ fo ] = getBkwdHandle( fi, opts )
if nargin < 2, opts = []; end
fo = @(layer, resi, resi_next) Backward(layer, resi, resi_next, fi, opts);

% -------------------------------------------------------------------------
function [ resi ] = Backward(layer, resi, resi_next, fi, opts)
if isempty(opts)
    resi.dzdx = fi(resi.x, layer.class, resi_next.dzdx);
else
    resi.dzdx = fi(resi.x, layer.class, resi_next.dzdx, opts);
end
