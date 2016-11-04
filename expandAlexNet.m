function [net] = expandAlexNet(net, varargin)
% EXPANDALEXNET   basically, construct a graphnn from the AlexNet or VGG-16 by expanding the network dimensions.
%   Options:
%     See code comments
% 
% Authors: Zhizhong Li
% 
% See the COPYING file.

assert(~isfield(net, 'paths'));
opts.mode = 'multiclass'; % type of last layer of the new path
opts.newtaskdim = 20; % # output of last layer
opts.extrawidth = 4096; % # neurons added to each layer
opts.init_scale = 0.01; % scale for randomly initializing new neurons
opts.weightInitMethod = 'gaussian'; % when randomly initializing, gaussian (fixed scale) or glorot (xavier, induce scale based on input/output of the layer)
opts.init_bias = 0; % what to init the biases as
opts.expand_layers = 3;
opts.expand_weights_source = 'net2net'; % how to initialize weights for new nodes?
opts.orig_loss = 'for_training'; % for_training: softmaxloss, for_eval: softmax, for_keep: custom, else: don't touch
opts.keep_response_loss = 'MI'; % MI for mutual information, L1 for L1-norm. only works when orig_loss is 'for_keep'.
opts.distillation_temp = 2; % only works when keepresponse_method is 'MI'.
opts.origfc_adddropout = true;
opts = vl_argparse(opts, varargin);

networkver = isfield(net, 'paths'); % 1 for vl_graphnn, 0 for vl_simplenn
assert(networkver == 0, 'expanding already expanded network -- not yet implemented');
% modify net to add dropouts... because the given networks do not have them
if opts.origfc_adddropout
    n_layers_orig = numel(net.layers);
    assert(n_layers_orig == 21 || n_layers_orig == 37);
    net.layers{end+1} = struct('type', 'dropout', 'rate', 0.5) ;
    net.layers{end+1} = struct('type', 'dropout', 'rate', 0.5) ;
    % net.layers = net.layers([ 1:15, 16,17,22, 18,19,23, 20,21 ]); % for AlexNet
    net.layers = net.layers([ 1:(n_layers_orig-6), n_layers_orig + [-5,-4,1, -3,-2,2, -1,0] ]); % AlexNet and VGG
end

n_layers_orig = numel(net.layers);
new_layers = 1; % only copy softmax weights...

% determine num of copied-over new layers
n_layers_conv = 0;
n_layers_expand = 0;
n_layers_copied = 0;
for i=1:n_layers_orig
    if strcmp(net.layers{end-i+1}.type, 'conv')
        n_layers_conv = n_layers_conv + 1;
    end
    if n_layers_conv == new_layers, n_layers_copied = i; end
    if n_layers_conv == opts.expand_layers, n_layers_expand = i; end
    if n_layers_expand && n_layers_copied
        break;
    end
end

% expand old layers
layers_hasweight = cellfun(@(X) isfield(X, 'weights'), net.layers);
layers_expanding = find(layers_hasweight & ((1:numel(layers_hasweight)) >= n_layers_orig-n_layers_expand+1));
net2net_map_last = []; net2net_map_this = [];
for i=1:numel(layers_expanding)
    i_layers = layers_expanding(i);
    % analyze old weights
    origweights = net.layers{i_layers}.weights;
    expdweights = origweights;
    assert(numel(origweights)==2);
    os = size(origweights{1});

    % pad 0 for new in to old out
    if i > 1
        expdweights{1} = cat(3, origweights{1}, zeros([os(1:2) opts.extrawidth os(4)], 'single'));
        os = size(expdweights{1});
    end
    % rand init new weights
    if i < numel(layers_expanding)
        switch opts.expand_weights_source
            case 'random'
                randnewweights = initWeights([os(1:3) opts.extrawidth], opts.init_scale, opts.init_bias, opts.weightInitMethod);
            case 'net2net'
                assert(opts.extrawidth <= os(4));
                net2net_map_this = randperm(os(4), opts.extrawidth);
                randnewweights = origweights{1}(:,:,:,net2net_map_this); % copy weights from existing nodes
                if ~isempty(net2net_map_last)
                    % evenly distributes weights between copies of last-layer nodes
                    randnewweights = cat(3, randnewweights, 0.5*origweights{1}(:,:,net2net_map_last,net2net_map_this));
                    randnewweights(:,:,net2net_map_last,:) = 0.5*randnewweights(:,:,net2net_map_last,:);
                end
                randnewweights = {randnewweights, origweights{2}(:,net2net_map_this)};
                net2net_map_last = net2net_map_this;
        end
        net.layers{i_layers}.oldtaskmask = reshape([ 1*ones(os(4),1); 0*ones(opts.extrawidth, 1) ], 1, 1, 1, []);
        expdweights{1} = cat(4, expdweights{1}, randnewweights{1});
        expdweights{2} = cat(2, origweights{2}, randnewweights{2});
    else
        % nothing actually happens here since the new layers are not shared by new/old tasks
    end
    net.layers{i_layers}.weights = expdweights;
    clear origweights expdweights randnewweights 
end

% copy layers
net.layers(end+1:end+n_layers_copied) = net.layers(end-n_layers_copied+1:end);
net.layers{end}.type = 'softmaxloss';
net.layers{end}.name = 'newtaskloss';

% brand new layers
n_layers_new = n_layers_copied;

% change new task loss and # classes
switch opts.mode
    case {'multiclass', 'multiclasshinge'}
        new_fc_last = n_layers_orig+n_layers_new-1;
        for i = 1:numel(net.layers{new_fc_last}.weights)
            weight_size = size(net.layers{new_fc_last}.weights{i});
            weight_size(end) = opts.newtaskdim;
            net.layers{new_fc_last}.weights{i} = zeros(weight_size, 'single');
        end
        % type defaults to 'softmaxloss', according to what we did when copying layers
        if strcmp(opts.mode, 'multiclasshinge')
            net.layers{end}.type = 'custom';
            net.layers{end}.name = 'multiclasshinge';
            layeropts.loss = 'mshinge';
            net.layers{end}.forward = getFwdHandle( @vl_nnloss_future, layeropts );
            net.layers{end}.backward = getBkwdHandle( @vl_nnloss_future, layeropts );
        end
    case 'multilabel'
        new_fc_last = n_layers_orig+n_layers_new-1;
        for i = 1:numel(net.layers{new_fc_last}.weights)
            weight_size = size(net.layers{new_fc_last}.weights{i});
            weight_size(end) = opts.newtaskdim;
            net.layers{new_fc_last}.weights{i} = zeros(weight_size, 'single');
        end
        net.layers{end}.type = 'custom';
        net.layers{end}.name = 'multilogreg';
        net.layers{end}.forward = getFwdHandle( @vl_nnlogregloss );
        net.layers{end}.backward = getBkwdHandle( @vl_nnlogregloss );
    otherwise
end
% rand init (not sure what to do about other fields e.g. momentum)
for i_layers = (n_layers_orig+1):(n_layers_orig+n_layers_new)
    if ~isfield(net.layers{i_layers}, 'weights'), continue; end
    net.layers{i_layers}.weights = initWeights(net.layers{i_layers}.weights, opts.init_scale, opts.init_bias, opts.weightInitMethod);
end

net.paths = {1:n_layers_orig, [ 1:(n_layers_orig-n_layers_copied), n_layers_orig+(1:n_layers_new)]};

% deal with different loss
switch opts.orig_loss
    case 'for_training'
        net.layers{n_layers_orig}.type = 'softmaxloss';
        net.layers{n_layers_orig}.name = 'loss';
    case 'for_eval'
        net.layers{n_layers_orig}.type = 'softmax';
        net.layers{n_layers_orig}.name = 'prob';
    case 'for_keep'
        % task "#1"
        net.layers{n_layers_orig}.type = 'custom';
        net.layers{n_layers_orig}.name = 'keepold';
        layeropts.temperature = opts.distillation_temp;
        layeropts.mode = opts.keep_response_loss;
        net.layers{n_layers_orig}.forward = getFwdHandle( @vl_nnsoftmaxdiff, layeropts );
        net.layers{n_layers_orig}.backward = getBkwdHandle( @vl_nnsoftmaxdiff, layeropts );
        % insert before task #1
        net.layers{end+1}.type = 'softmaxloss';
        net.layers{end}.name = 'origloss';
        net.paths = { [ 1:n_layers_orig-1, numel(net.layers) ], net.paths{:} };
end

if strcmp(opts.orig_loss, 'for_keep')
end


% -------------------------------------------------------------------------
function [weights] = initWeights(weights, scale, bias, method)
%% initWeights: random re-initialize weights
% e.g. net.layers{i_layers}.weights = initWeights(net.layers{i_layers}.weights, opts.init_scale, opts.init_bias, opts.weightInitMethod)

if ~iscell(weights) && isvector(weights) && numel(weights)==4 % directly given the weight sizes
    sz = weights;
elseif ~iscell(weights) % given just weights not bias
    sz = size(weights);
else % both weight and bias
    sz = size(weights{1});
end

% determine actual scale
switch method
    case 'gaussian'
        sc = scale ;
    case 'glorot'
        sc = sqrt(1/(sz(1)*sz(2)*(sz(3)+sz(4))/scale)) ;
    otherwise
        throw(MException('forkAlexNet:weightInitMethod','Unrecognized weightInitMethod value'));
end

if ~iscell(weights) && isvector(weights) && numel(weights)==4 % directly given the weight sizes
    weights = cell(1,2);
    weights{1} = single(sc) * randn(sz, 'single');
    weights{2} = single(bias) + zeros([1 sz(4)], 'single');
elseif ~iscell(weights) % given old weight no bias
    weights = sc * ...
        randn(sz, 'single');
else
    % only deals with "weight+bias combination"
    assert(numel(weights) == 2);
    weights{1} = single(sc) * randn(size(weights{1}), 'single');
    weights{2} = single(bias) + zeros(size(weights{2}), 'single');
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
