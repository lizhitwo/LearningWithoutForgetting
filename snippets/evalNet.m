function [ info ] = evalNet( dset_name, net_mod, testPath, varargin )
% EVALNET  get the evaluation softmax's of a trained net on some data
% and the top1 / top5 count for each class
%   Input:
%     DSET_NAME name of the dataset that is evaluated on
%     NET_MOD a modified (fine-tune/LwF/etc.) network that is being evaluated.
%       Can be a string (.mat file name), a struct (with .layers, .paths, ...),
%       or a [] which defaults to the AlexNet trained on ImageNet.
%     TESTPATH the root (input) to leaf (loss) path on the network to evaluate
%   Output:
%     INFO structure for evaluation results.
%   Options:
%     See code comments

% Authors: Zhizhong Li
% 
% See the COPYING file.

opts.getLastFCOutput = true; % Defaults to include the penultimate layer response in the INFO output.
opts.errorFunction = []; % usually figured out automatically from DSET_NAME; overwrite at your own risk
opts.errorLabels = []; % usually figured out automatically from DSET_NAME; overwrite at your own risk
opts.augment = false; % Whether to use data augmentation at test time. true defaults to 'f5'
opts.partial_label = []; % Evaluate on only part of the labels. NOTE: only when evaluating. Using this code to get last_fc should include all samples, due to the shuffling random seed issue.
opts.set_gt_to_1 = false; % When you do not have ground truth (say, recording response for LwF, or evaluating on VOC test set), pass a dummy label.
opts = vl_argparse(opts, varargin) ;

% path and flags
p = getPath('path_dump', '../dump/');

% maybe load a pre-trained (or fine-tuned, etc.) CNN from disk
if ischar(net_mod)
    net = load(net_mod) ;
    net = net.net;
    if isfield(net, 'paths')
        taskPath = testPath;
    else
        disp('WARNING: vl_simplenn detected; assuming evaluation on original network. Replacing some layers with the original network');
        orig_net = load(p.file_caffe_alex) ;
        redo_layers = [-1 -3 -5] 
        keyboard;
        for iter_layers = 1:numel(redo_layers)
            i_layers = redo_layers(iter_layers);
            net.layers{end+i_layers} = orig_net.layers{end+i_layers};
        end
        if isfield(net, 'oldlast'), net = rmfield(net, 'oldlast'); end
        clear orig_net
        net.layers{end}.type = 'softmaxloss';
        net.layers{end}.class = [];
        taskPath = [1];
    end
elseif isstruct(net_mod)
    net = net_mod ; 
    taskPath = testPath;
else
    net = load(p.file_caffe_alex) ;
    net.layers{end}.type = 'softmaxloss';
    taskPath = [1];
end





% load image metadata; specify dataset metadata
imdb.images.name = [];
imdb.images.label = [];
imdb.images.set = [];
imdb.images.task = [];
errorFunction = [];
errorLabels = [];
switch dset_name
    case 'ILSVRC2012_ev'
        imdb = IMDBAddImageNet( imdb, p, 1 );
        imgroot = p.path_imgtrain;
    case 'ILSVRC2012_tr'
        imdb = IMDBAddImageNet( imdb, p, 1 );
        imgroot = p.path_imgtrain;
        imdb.images.set = 3 - imdb.images.set;
    case 'VOC2012_ev'
        imdb = IMDBAddVOCPerson( imdb, p, 1, 'label', 'nonper/person' );
        imgroot = p.path_VOCimdir;
    case 'VOC2012_tr'
        imdb = IMDBAddVOCPerson( imdb, p, 1, 'label', 'nonper/person' );
        imgroot = p.path_VOCimdir;
        imdb.images.set = 3 - imdb.images.set;
    case 'VOC2012_c20_ev'
        imdb = IMDBAddVOCPerson( imdb, p, 1, 'label', 'multilabel' );
        imgroot = p.path_VOCimdir;
        errorFunction = 'multilabel';
        errorLabels = {'eval_top1e', 'eval_mock_top5e' };
    case 'VOC2012_c20_tr'
        imdb = IMDBAddVOCPerson( imdb, p, 1, 'label', 'multilabel' );
        imgroot = p.path_VOCimdir;
        imdb.images.set = 3 - imdb.images.set;
        errorFunction = 'multilabel';
        errorLabels = {'eval_top1e', 'eval_mock_top5e' };
    case 'VOC2012_trval'
        imdb = IMDBAddVOCtrval( imdb, p, 1, 'label', 'nonper/person' );
        imgroot = p.path_VOCimdir;
        imdb.images.set = 3 - imdb.images.set;
    case 'VOC2012_c20_test'
        imdb = IMDBAddVOCtrval( imdb, p, 1, 'label', 'multilabel' );
        imgroot = p.path_VOCimdir;
        errorFunction = 'multilabel';
        errorLabels = {'eval_top1e', 'eval_mock_top5e' };
    case 'Places2_tr_1cls'
        imdb = IMDBAddPlaces2( imdb, p, 1 );
        imgroot = p.path_Placesroot;
        sel = cell2mat(imdb.images.label)==10;
        for f = fieldnames(imdb.images)', imdb.images.(f{1}) = imdb.images.(f{1})(sel); end
        imdb.images.set = 3 - imdb.images.set;
    case 'Places2_ev'
        imdb = IMDBAddPlaces2( imdb, p, 1 );
        imgroot = p.path_Placesroot;
    case 'Places2_test'
        imdb = IMDBAddPlaces2( imdb, p, 1, 'trainval', [3] );
        imdb.images.set(:) = 2;
        imgroot = p.path_Placesroot;
    case 'CUB_ev'
        imdb = IMDBAddCUB( imdb, p, 1 );
        imgroot = p.path_CUBimdir;
    case 'CUB_tr'
        imdb = IMDBAddCUB( imdb, p, 1 );
        imgroot = p.path_CUBimdir;
        imdb.images.set = 3 - imdb.images.set;
    case 'MIT67_ev'
        imdb = IMDBAddMIT67( imdb, p, 1 );
        imgroot = p.path_MIT67imdir;
    case 'MIT67_tr'
        imdb = IMDBAddMIT67( imdb, p, 1 );
        imgroot = p.path_MIT67imdir;
        imdb.images.set = 3 - imdb.images.set;
    case 'MNIST_ev'
        imdb = IMDBAddMNIST( imdb, p, 1 );
        imgroot = p.path_MNISTimdir;
    case 'MNIST_tr'
        imdb = IMDBAddMNIST( imdb, p, 1 );
        imgroot = p.path_MNISTimdir;
        imdb.images.set = 3 - imdb.images.set;
    otherwise
        throw(MException('dset_name:unknown', 'Unknown dataset specified.'));
end

if ~isempty(opts.partial_label)
    imdb = trimIMDBforBatch(imdb, 1, opts.partial_label);
end

if opts.set_gt_to_1
    [imdb.images.label{imdb.images.set==2}] = deal(1);
end

% use opts.errorFunction and opts.errorLabels if they are not empty.
% otherwise, use the variables errorFunction and errorLabels from above if they are not empty.
% otherwise, remove the fields from OPTS to use the default value 
if isnumeric(opts.errorFunction) && isempty(opts.errorFunction) % nothing specified from varargin
  if isnumeric(errorFunction) && isempty(errorFunction) % nothing specified due to dataset
    opts = rmfield(opts, 'errorFunction');
  else % something specified due to dataset
    opts.errorFunction = errorFunction;
  end
end % else use whatever is provided
if isnumeric(opts.errorLabels) && isempty(opts.errorLabels) % nothing specified from varargin
  if isnumeric(errorLabels) && isempty(errorLabels) % nothing specified due to dataset
    opts = rmfield(opts, 'errorLabels'); % use default from cnn_customeval
  else % something specified due to dataset
    opts.errorLabels = errorLabels;
  end
end % else use whatever is provided


% get batch options
bopts = net.normalization ;
bopts.numThreads = 8 ;

if opts.augment
    assert(~isfield(opts, 'errorFunction') || strcmp(opts.errorFunction, 'multiclass'));
    if isstr(opts.augment)
        switch opts.augment
        case 'f5'
            opts.errorFunction = 'multiclass_aug_f5';
            bopts.numAugments = 10 ; %augcrop%
            bopts.transformation = 'f5' ; %augcrop%
            opts.batchSize = 64;
        case 'f6'
            opts.errorFunction = 'multiclass_aug_f6';
            bopts.numAugments = 12;
            bopts.transformation = 'f6';
            opts.batchSize = 64;
        end
    end
end
opts = rmfield(opts, 'augment');
opts = rmfield(opts, 'partial_label');
opts = rmfield(opts, 'set_gt_to_1');

% or else no jittering whatsoever!

fn = getBatchWrapper(bopts, imgroot);

% wrapper that does evaluation using CNN_CUSTOMTRAIN using some default settings
info = cnn_customeval(net, imdb, fn, taskPath, p.path_dump, opts );
info.val.label = imdb.images.label(imdb.images.set==2);



function [ info ] = cnn_customeval(net, imdb, getBatch, taskPath, path_dump, varargin )% no opts
% wrapper that does evaluation using CNN_CUSTOMTRAIN using some default settings

% training core params
opts.train.numEpochs = 1;
opts.train.expDir = path_dump ;
opts.train.errorFunction = 'multiclass';

% for training details
opts.train.batchSize = 192 ;
opts.train.numSubBatches = 1 ;
opts.train.continue = true ;
opts.train.gpus = [1] ;
opts.train.prefetch = true ;
opts.train.sync = true ;
opts.train.cudnn = true ;

% options for fork-training
opts.train.errorLabels = {'eval_top1e', 'eval_top5e' };
opts.train.taskPaths = taskPath;
opts.train.majorTask = 1;

opts.train.learningRate = 0;% logspace(-2, -4, 60) ;
opts.train.train = nan;
opts.train.getLastFCOutput = true;

opts.train = vl_argparse(opts.train, varargin);


[~,info] = cnn_customtrain(net, imdb, getBatch, opts.train, 'conserveMemory', true, 'figure_interval', false) ;
