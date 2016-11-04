function [ net, info ] = PASCALkeeporig( varargin )
% PASCALKEEPORIG   Given pretrained model, perform LwF or baseline on new dataset. (Not just PASCAL)
% 
%   Output:
%     NET network and INFO evaluation results
%   Options:
%     See code comments
% 
% Authors: Zhizhong Li
% 
% See the COPYING file.

% path and flags
opts.path_dump = '../dump/'; % directory for dumping results
opts.keep_orig = 1; % 0: fine-tune; 1: LwF keep response; 2: keep shared layers untouched; 3: joint train; 4: L2 constraint to orig
opts.fakesave = false; % if true, then instead of saving the training progress, a dummy save file is created to save space.

opts.orignet = 'Alex_ImNet'; % string codename for the original network and dataset
opts.newtask = 'VOC'; % string codename for the new dataset
opts.change_struct = 'fork'; % how to change network structure for the new task. fork: tree-structure, expand: network expansion
opts.origfc_adddropout = true; % whether or not add dropout to the original network.
opts.fork_layers = 2; % number of old fc layers to treat as task-specific
opts.redo_layers = 2; % number of layers to re-initialize
opts.new_layers = 0; % number of layers to add in addition to what's already there.
opts.new_layers_scale = 1; % param initialization Gaussian scale (in addition to using glorot, therefore smaller than this and varies by task) 
opts.keep_response_loss = 'MI'; % MI for cross-entropy/knowledge distillation. L1 for L1, L2 for L2. 
opts.distillation_temp = 2; % only works when keepresponse_method is MI. specifies the temperature for knowledge distillation.
opts.weightInitMethod = 'glorot'; % use glorot to automatically figure out the weight initialization method, or use gaussian to directly specify the std
opts.learningRate = 0.002; % base learning rate. Use a scalar.
opts.sharedWeightLRmultiplier = 1; % learning rate multiplier for shared parameters
opts.numEpochs = 40; % number of epochs
opts.keep_response_lambda = 1; % the lambda for the response preserving loss
opts.lr_structure = [1, 1, 1, 0.1]; % The shape of learning rate schedule. This will be multiplied with the learning rate and stretched to NUMEPOCHS length.
opts.partial_traintest = [1, 0.3]; % Permanently subsample the [train, val] dataset for this experiment.
opts.partial_randseed = randi(intmax()); % the seed for dataset subsampling randomness. Can be specified (fixed) for each irun. All datasets use the same seed since their #samples are different
opts.accumulative = []; % when cumulative adding new tasks, specify the .mat file for the last network (w/ one less new task) here.
opts.accumu_batches = {}; % when cumulative adding new tasks, specify the classes separated for each chunk here.

opts = vl_argparse(opts, varargin);
assert(opts.keep_orig >=0 && opts.keep_orig <=4 && floor(opts.keep_orig)==opts.keep_orig);

% get path strings
p = getPath('path_dump', opts.path_dump);


% -------------------------------------------------------------------------
% Specification of old task & new task(s)
% -------------------------------------------------------------------------

% figure out the specifications of the original network
switch opts.orignet
    case {'Alex_ImNet', 'VGG_ImNet'}
        % pretrained network file
        if strcmp(opts.orignet, 'Alex_ImNet'), net = (p.file_caffe_alex) ;
        elseif strcmp(opts.orignet, 'VGG_ImNet'), net = (p.file_caffe_vgg) ;
        end
        % cache for response on the new dataset
        p.path_OrigNetresponse = p.path_response.(opts.orignet);
        % original task specs
        oldtaskopts.name = 'ImNet';   oldtaskopts.path_origtask = p.path_imgtrain; % image path
        oldtaskopts.evalmode = 'ILSVRC2012_ev'; % string for calling evalNet()
        oldtaskopts.IMDBAddOldtask = @IMDBAddImageNet; % how to add the dataset to imdb
        oldtaskopts.errorFN = 'multiclass';
    case 'Alex_Places2'
        % pretrained network file
        net = (p.file_caffe_alex_places2) ;
        % cache for response on the new dataset
        p.path_OrigNetresponse = p.path_response.(opts.orignet);
        % original task specs
        oldtaskopts.name = 'Places2'; oldtaskopts.path_origtask = p.path_Placesroot; % image path
        oldtaskopts.evalmode = 'Places2_ev'; % string for calling evalNet()
        oldtaskopts.IMDBAddOldtask = @IMDBAddPlaces2; % how to add the dataset to imdb
        oldtaskopts.errorFN = 'multiclass';
    otherwise
        throw(MException('opts:value', 'opts.orignet unrecognized'));
end
% memory issues -- smaller batch for VGG
if strcmp(opts.orignet, 'VGG_ImNet'), batchSize=64;
else batchSize=192;
end

% cumulative adding new tasks mode.
accum_mode = ~isempty(opts.accumulative);
if accum_mode 
    % whether to use specified pre-trained model
    if ~strcmp(opts.accumulative, 'original')
        net = opts.accumulative;  
    end
end

% load the pre-trained CNN from disk, and clear the network of training state.
net = load(net); if ~isfield(net, 'layers'), net = net.net; end
net = resetNetMomentum(net, 'fields', {'momentum', 'momentum2', 'updateCount', 'learningRate'}, 'mode', 'remove');


% specifications parameters for the new task
switch opts.newtask
        % string name for printing/plotting; string to pass to evalNet() when recording response;
        % new loss layer form;  new last-layer number of nodes; how to add the dataset to imdb;
        % the label form to pass to IMDBAdd*;   image files root;   error function when evaluating;
        % string to pass to evalNet() when evaluating;
    case 'VOC'
        newtaskopts.name = 'PASCAL';    newtaskopts.response_evalset = 'VOC2012_tr';
        newtaskopts.forkmode = 'multilabel'; newtaskopts.newtaskdim = 20;  newtaskopts.IMDBAddNewtask = @IMDBAddVOCPerson;
        newtaskopts.imdblabelmode = 'multilabel'; newtaskopts.path_imdir = p.path_VOCimdir; newtaskopts.errorFN = 'multilabel';
        newtaskopts.evalmode = 'VOC2012_c20_ev';
    case 'VOCtrval'
        newtaskopts.name = 'PASCALtrval';    newtaskopts.response_evalset = 'VOC2012_trval';
        newtaskopts.forkmode = 'multilabel'; newtaskopts.newtaskdim = 20;  newtaskopts.IMDBAddNewtask = @IMDBAddVOCtrval;
        newtaskopts.imdblabelmode = 'multilabel'; newtaskopts.path_imdir = p.path_VOCimdir; newtaskopts.errorFN = 'multilabel';
        newtaskopts.evalmode = 'VOC2012_c20_test';
    case 'CUB'
        newtaskopts.name = 'CUB_birds'; newtaskopts.response_evalset = 'CUB_tr';
        newtaskopts.forkmode = 'multiclass'; newtaskopts.newtaskdim = 200; newtaskopts.IMDBAddNewtask = @IMDBAddCUB;
        newtaskopts.imdblabelmode = 'class';      newtaskopts.path_imdir = p.path_CUBimdir; newtaskopts.errorFN = 'multiclass';
        newtaskopts.evalmode = 'CUB_ev';
        % hinge trial
        % C = 0.1; newtaskopts.forkmode = 'multiclasshinge'; newtaskopts.outputlayerdecay = 1/C;
    case 'MIT67'
        newtaskopts.name = 'MIT67_scene'; newtaskopts.response_evalset = 'MIT67_tr';
        newtaskopts.forkmode = 'multiclass'; newtaskopts.newtaskdim = 67; newtaskopts.IMDBAddNewtask = @IMDBAddMIT67;
        newtaskopts.imdblabelmode = 'class';      newtaskopts.path_imdir = p.path_MIT67imdir; newtaskopts.errorFN = 'multiclass';
        newtaskopts.evalmode = 'MIT67_ev';
    case 'MNIST'
        newtaskopts.name = 'MNIST'; newtaskopts.response_evalset = 'MNIST_tr'; % keyboard;
        newtaskopts.forkmode = 'multiclass'; newtaskopts.newtaskdim = 10; newtaskopts.IMDBAddNewtask = @IMDBAddMNIST;
        newtaskopts.imdblabelmode = 'class';      newtaskopts.path_imdir = p.path_MNISTimdir; newtaskopts.errorFN = 'multiclass';
        newtaskopts.evalmode = 'MNIST_ev';
    otherwise, throw(MException('opts:value', 'opts.newtask unrecognized'));
end


% for accumulating new task... derive the specifications for the 'new old tasks'
accum_n_oldtasks = 1;
if accum_mode
    % determine how many old tasks there are
    if isfield(net, 'paths'), accum_n_oldtasks = (numel(net.paths)+1)/2; end % (otherwise, surely there is only one old task, the same as before.)
    % identities of the old tasks (add _1 ~ _3 suffix)
    accum_names = strcat(newtaskopts.name, '_', arrayfun(@(X) sprintf('%d',X), 1:(accum_n_oldtasks-1), 'UniformOutput', false));
    % derive the extra 'old task' specs from the new task spec (since they use the same dataset)
    [oldtaskopts(2:accum_n_oldtasks).name] = deal(accum_names{:});
    [oldtaskopts(2:accum_n_oldtasks).path_origtask] = deal(newtaskopts.path_imdir);
    [oldtaskopts(2:accum_n_oldtasks).evalmode] = deal(newtaskopts.evalmode);
    [oldtaskopts(2:accum_n_oldtasks).IMDBAddOldtask] = deal(newtaskopts.IMDBAddNewtask);
    [oldtaskopts(2:accum_n_oldtasks).errorFN] = deal(newtaskopts.errorFN);
    accum_newtask_batch = accum_n_oldtasks;
    assert(sum(cellfun(@numel, opts.accumu_batches)) == newtaskopts.newtaskdim);
    newtaskopts.newtaskdim = numel(opts.accumu_batches{accum_newtask_batch}); 
end


% -------------------------------------------------------------------------
% Recording response of old model on new task
% -------------------------------------------------------------------------
if ~accum_mode 
    % normally, 1 old task 1 new task, everything is cached in cache folder
    if ~exist(p.path_OrigNetresponse.(opts.newtask).train, 'file')
        fprintf('Pre-calculating response of %s on %s.\n', opts.orignet, opts.newtask);
        % prepare the model for recording response and run through dataset
        bkuptyp = net.layers{end}.type;    net.layers{end}.type = 'softmaxloss';
        info = evalNet( newtaskopts.response_evalset, net, 1, 'getLastFCOutput', true );
        lastfc_out = info.val.lastfc_out{1};
        % reshape response, pass through softmax, and save
        sz = size(lastfc_out); lastfc_out = reshape(vl_nnsoftmax(reshape(lastfc_out, [1 1 sz])), sz);
        save(p.path_OrigNetresponse.(opts.newtask).train, 'lastfc_out');
        net.layers{end}.type = bkuptyp;
    end
else 
    % for accum_mode, cache in the dump_dir, because the starting network depends on how it was trained for last chunk.
    accum_responsefile = fullfile(p.path_dump, 'oldtask_response.mat');
    if exist(accum_responsefile, 'file')
        accum_response = load(accum_responsefile); accum_response = accum_response.accum_response;
    else
        accum_response = cell(accum_n_oldtasks, 1);
        for i=1:accum_n_oldtasks
            fprintf('Pre-calculating response of task %s on %s.\n', oldtaskopts(i).name, opts.newtask);
            % prepare the model for recording response and run through dataset
            if isfield(net, 'paths'), accum_lastlayer = net.paths{i*2-1}(end);
            else accum_lastlayer = numel(net.layers);
            end
            bkuptyp = net.layers{accum_lastlayer}.type;    net.layers{accum_lastlayer}.type = 'softmaxloss'; % a dummy loss that does not explode when fed with integers as the label. we only need the logits
            info = evalNet( newtaskopts.response_evalset, net, i*2-1, 'getLastFCOutput', true, 'set_gt_to_1', true );
            lastfc_out = info.val.lastfc_out{1};
            % reshape response, pass through softmax, and save
            switch oldtaskopts(i).errorFN
                case 'multiclass', accum_normfn = @vl_nnsoftmax;
                case 'multilabel', accum_normfn = @vl_nnsigmoid;
            end
            sz = size(lastfc_out); lastfc_out = reshape(accum_normfn(reshape(lastfc_out, [1 1 sz])), sz);
            accum_response{i} = lastfc_out;
            net.layers{accum_lastlayer}.type = bkuptyp;
        end
        save(accum_responsefile, 'accum_response');
        clear lastfc_out
    end
end

% -------------------------------------------------------------------------
% EDIT NET to accomodate the new task so that it's ready for retraining 
% -------------------------------------------------------------------------
switch opts.change_struct
    case 'fork' % tree-structured
        net = forkAlexNet(net, opts.fork_layers, ... % fork starting from which layer
            'mode', newtaskopts.forkmode, ...
            'newtaskdim', newtaskopts.newtaskdim, ...
            'init_scale', opts.new_layers_scale, ...
            'weightInitMethod', opts.weightInitMethod, ...
            'init_bias', 0, ...
            'redo_layers', opts.redo_layers, ...
            'new_layers', opts.new_layers, ...
            'origfc_adddropout', opts.origfc_adddropout && (accum_n_oldtasks<=1), ... % if accum_mode, the dropouts are already added the first time.
            'orig_loss', 'for_keep', ...
            'keep_response_loss', opts.keep_response_loss, ...
            'distillation_temp', opts.distillation_temp);

    case 'expand' % network expansion
        assert(opts.fork_layers == 3);
        assert(opts.redo_layers == 1);
        assert(~accum_mode);
        net = expandAlexNet(net, ...
            'mode', newtaskopts.forkmode, ...
            'newtaskdim', newtaskopts.newtaskdim, ...
            'extrawidth', 1024, ...
            'init_scale', opts.new_layers_scale, ...
            'weightInitMethod', opts.weightInitMethod, ...
            'init_bias', 0, ...
            'expand_layers', opts.fork_layers, ...
            'origfc_adddropout', opts.origfc_adddropout, ...
            'orig_loss', 'for_keep', ...
            'keep_response_loss', opts.keep_response_loss, ...
            'distillation_temp', opts.distillation_temp);

    otherwise
        throw(MException('opts:value', 'opts.orignet unrecognized'));
end

% unused experiment: hinge loss + L2 regularization for last fc
if isfield(newtaskopts, 'outputlayerdecay')
    new_outputweightlayer = net.paths{end}(end-1);
    assert(isfield(net.layers{new_outputweightlayer}, 'weights'));
    net.layers{new_outputweightlayer}.weightDecay = [ newtaskopts.outputlayerdecay, 0 ];
end


% -------------------------------------------------------------------------
% Adding datasets to IMDB
% -------------------------------------------------------------------------

randseed = opts.partial_randseed;
% initialize image metadata
imdb.images.name = [];
imdb.images.label = [];
imdb.images.set = [];
imdb.images.task = [];

% for the very first task -- special treatment
% original task -- val or train&val?
if opts.keep_orig == 3, trainval = [1 2]; % joint train
else trainval = [2]; % just test it; do not joint train
end
imdb = oldtaskopts(1).IMDBAddOldtask( imdb, p, 1, 'trainval', trainval, 'randstream', RandStream('mt19937ar', 'Seed', randseed) );

% the LwF keeping response of the very first task.
if opts.keep_orig == 1 % keep old response
    % the path p is edited so that p.path_OrigNetresponse.(opts.newtask) is the corresponding net's response on the new task.
    % Note: keeping response on the old task using new data.
    if ~accum_mode
        [imdb] = newtaskopts.IMDBAddNewtask( imdb, p, 2, 'trainval', [1], 'label', 'probdist', 'partial', opts.partial_traintest, 'randstream', RandStream('mt19937ar', 'Seed', randseed) ); % *CHANGE*
    else % accum_mode
        % if the new task is split 3-ways (less tr data), then newtask keep original images should be split 3-ways too
        [imdb, select] = newtaskopts.IMDBAddNewtask( imdb, p, 2, 'trainval', [1], 'partial', opts.partial_traintest, 'randstream', RandStream('mt19937ar', 'Seed', randseed) ); % *CHANGE*
        [imdb, isin] = trimIMDBforBatch(imdb, 2  , opts.accumu_batches{accum_newtask_batch});
        % substitute with recorded response
        probdist = accum_response{1}(:,select.train(isin));
        imdb.images.label(imdb.images.task==2) = num2cell( probdist, 1 )';
    end
end

% for the other 'old tasks' in cumulative new task mode
if accum_mode
    % ASSERT: i_task - 1 == i_batch
    assert(numel(net.paths) == (1+accum_n_oldtasks)*2 - 1);
    for i=2:accum_n_oldtasks
        % add old task val / train&val
        imdb = oldtaskopts(i).IMDBAddOldtask( imdb, p, i*2-1, 'trainval', trainval, 'randstream', RandStream('mt19937ar', 'Seed', randseed) );
        imdb = trimIMDBforBatch(imdb, i*2-1, opts.accumu_batches{i-1});
        % response for LwF
        if opts.keep_orig == 1 % keep old response
            % if the new task is split 3-ways (less tr data), then newtask keep original images should be split 3-ways too
            [imdb, select] = newtaskopts.IMDBAddNewtask( imdb, p, i*2  , 'trainval', [1], 'partial', opts.partial_traintest, 'randstream', RandStream('mt19937ar', 'Seed', randseed) );
            [imdb, isin] = trimIMDBforBatch(imdb, i*2  , opts.accumu_batches{accum_newtask_batch});
            % substitute with recorded response
            probdist = accum_response{i}(:,select.train(isin));
            imdb.images.label(imdb.images.task==i*2) = num2cell( probdist, 1 )';
        end
    end
end

% for the final new task -- special treatment
% just add the dataset and trim
imdb = newtaskopts.IMDBAddNewtask( imdb, p, accum_n_oldtasks*2+1, 'trainval', [1 2], 'label', newtaskopts.imdblabelmode, 'partial', opts.partial_traintest, 'randstream', RandStream('mt19937ar', 'Seed', randseed) );
if accum_mode
    imdb = trimIMDBforBatch(imdb, accum_newtask_batch*2+1, opts.accumu_batches{accum_newtask_batch});
end


% -------------------------------------------------------------------------
% Data batch settings
% -------------------------------------------------------------------------

% get batch options
bopts = net.normalization ;
bopts.numThreads = 8 ;
bopts.transformation = 'f25'; % using a 5x5 grid for random translation+cropping, and random flipping

% RGB variance for jittering. Obtained like this:
% batch = randperm(1281144, 10000); [ims, lbls] = fn(imdb,batch); % get random batch from dataset
% ims = permute(ims,[3 1 2 4]); ims = reshape(ims,3,[]);
% cov = ims * ims' / (size(ims,2)-1) - (mean(ims,2)*mean(ims,2)');
% [V,D] = eig(cov);
% V*sqrt(D)*.1
switch opts.orignet
    case {'Alex_ImNet', 'VGG_ImNet'}
        bopts.rgbVariance = [
           -0.6738   -2.5236    6.5172
            1.3706    0.0298    6.6587
           -0.7124    2.4444    6.6470
        ];
    case 'Alex_Places2'
        bopts.rgbVariance = [
            0.5842    2.3173    6.2728
           -1.1196    0.1126    6.5503
            0.5431   -2.2607    6.7561
        ];
    otherwise
        throw(MException('opts:value', 'opts.orignet unrecognized'));
end

% the getBatch function handles for each root(input)-to-leaf(loss) path.
fn = {};
for i=1:accum_n_oldtasks
    fn = [fn, ...
        {getBatchWrapper(bopts, oldtaskopts(i).path_origtask), ... % i-th old task's source images
         getBatchWrapper(bopts, newtaskopts.path_imdir)}];  % keep response on the new task's source images
end
fn = [fn, { getBatchWrapper(bopts, newtaskopts.path_imdir) }]; % the new task's source images


% -------------------------------------------------------------------------
% Training core parameters
% -------------------------------------------------------------------------
opts.train.numEpochs = opts.numEpochs;
opts.train.backPropDepth = +inf;
if opts.keep_orig == 2
    % locking shared parameters.
    opts.train.backPropDepth = 3 * (opts.fork_layers + opts.new_layers) - 1;
    switch opts.change_struct
        case 'expand'
            % special hacks for locking layers for network expansion, where
            % task-specific parameters are mingled into one "weights" field
            assert(~accum_mode);
            for i=1:numel(net.layers)
                if isfield(net.layers{i}, 'oldtaskmask')
                    lrmask = ~net.layers{i}.oldtaskmask;
                    net.layers{i}.learningRate = {lrmask, reshape(lrmask, 1, [])};
                end
            end
            % expand-lock: first lock added neurons, then release the lock
            bpd_unlock = opts.train.backPropDepth;
            bpd_lock = 3 * (opts.redo_layers + opts.new_layers) - 1;
            opts.train.backPropDepth = ones(opts.numEpochs,1)*bpd_unlock; % backprop depth for each epoch
            opts.train.backPropDepth(1:ceil(opts.numEpochs/4)) = bpd_lock;
    end
end
% Preserving parameter values by L2 loss?
if opts.keep_orig == 4
    opts.train.l2keepweightsDepth = numel(net.paths{end}) - (3 * (opts.fork_layers + opts.new_layers) - 1);
    opts.train.l2keepweightsDecay = opts.keep_response_lambda * 0.0005*100; % 100x as large as the normal weight decay, times lambda
    assert(strcmp(opts.change_struct, 'fork'));
    assert(opts.new_layers == 0);
end
% Lower learning rate for shared layers?
if opts.sharedWeightLRmultiplier ~= 1
    backpropdepth_lr_1 = 3 * (opts.fork_layers + opts.new_layers) - 1;
    n_layers_lr_small = numel(net.paths{end}) - backpropdepth_lr_1;
    switch opts.change_struct
        case 'fork'
            % ... by changing the layer-specific lr
            for i=net.paths{end}(1:n_layers_lr_small)
                assert(~isfield(net.layers{i}, 'filter'));
                if isfield(net.layers{i}, 'weights')
                    J = numel(net.layers{i}.weights) ;
                    net.layers{i}.learningRate = ones(1, J, 'single') * opts.sharedWeightLRmultiplier ;
                end
            end
        case 'expand'
            assert(false, 'Work In Progress'); % still to adjust according to oldtaskmask
    end

end

opts.train.expDir = p.path_dump ;


% assemble old tasks' opts.train.errorFunction (one for each path, but LwF paths just uses 'none')
opts.train.errorFunction = {};
for i=1:accum_n_oldtasks
    accum_errorFunc = {
        oldtaskopts(i).errorFN, 'none'; % training
        oldtaskopts(i).errorFN, 'none' % testing
    };
    if opts.keep_orig ~= 3 % joint train
        accum_errorFunc{1,1} = 'none';
    end
    opts.train.errorFunction = [ opts.train.errorFunction, accum_errorFunc ] ;
end
% new task's
opts.train.errorFunction = [ opts.train.errorFunction, {newtaskopts.errorFN; newtaskopts.errorFN} ];

% assemble old tasks' opts.train.errorLabels
opts.train.errorLabels = {{}, {}};
for i=1:accum_n_oldtasks
    ich = sprintf('%d',i-1); if i==1, ich = ' '; end
    % test: loss objective, error (1), error (top 5)
    opts.train.errorLabels{2} = [opts.train.errorLabels{2}, {['o:OrigTask' ich], ['OrigTask' ich '_top1e'], ['OrigTask' ich '_top5e(mock)']}];
    if opts.keep_orig == 1
        % train: LwF: just loss objective
        opts.train.errorLabels{1} = [ opts.train.errorLabels{1} opts.train.errorLabels{2}(end-2)];
    elseif opts.keep_orig == 3
        % train: joint train: loss objective, error (1), error (top 5)
        opts.train.errorLabels{1} = [ opts.train.errorLabels{1} opts.train.errorLabels{2}(end-2:end) ];
    end
end
% new task's (loss objective, error 1, error 5)
opts.train.errorLabels{1} = [ opts.train.errorLabels{1} { 'o:NewTask', 'NewTask_errorall', 'NewTask_top5e(mock)' }];
opts.train.errorLabels{2} = [ opts.train.errorLabels{2} { 'o:NewTask', 'NewTask_errorall', 'NewTask_top5e(mock)' }];
% replace with actual dataset names
for i=1:numel(opts.train.errorLabels), for j=1:numel(opts.train.errorLabels{i})
    opts.train.errorLabels{i}{j} = strrep(opts.train.errorLabels{i}{j}, 'OrigTask ', oldtaskopts(1).name);
    opts.train.errorLabels{i}{j} = strrep(opts.train.errorLabels{i}{j}, 'NewTask', newtaskopts.name);
end; end


% for training details
opts.train.batchSize = batchSize;
opts.train.numSubBatches = 1 ;
opts.train.continue = true ;
opts.train.gpus = [1] ;
opts.train.prefetch = true ;
opts.train.sync = false ;
opts.train.cudnn = true ;

% options added for multiple paths
opts.train.taskPaths = 1:numel(net.paths); % note that task paths with no imdb entries associated will just be skipped.
opts.train.majorTask = numel(net.paths);
opts.train.balanceLoss = [ repmat([1; opts.keep_response_lambda], accum_n_oldtasks, 1) ; 1]; % joint training does not take lambda; LwF does
opts.train.shuffleSyncTasks = [];
if opts.keep_orig == 1
    opts.train.shuffleSyncTasks = [2*(1:accum_n_oldtasks), 2*accum_n_oldtasks+1]; % use the same batch for all keep-response / new task training paths
end
opts.train.shuffleVal = true;
opts.train.getLastFCOutput = false;

opts.train.fakesave = opts.fakesave;

% learning rate schedule from the shape and base value
lr_structure = opts.lr_structure; lr_structure = lr_structure(floor((0:opts.numEpochs-1) / opts.numEpochs * numel(lr_structure) + 1));
opts.train.learningRate = opts.learningRate * lr_structure;% logspace(-3, -3, 20); % logspace(-3, -4, 40); % logspace(-3, -5, 20); *FULL* % logspace(-2, -4, 60) ;


% -------------------------------------------------------------------------
% Actual Training
% -------------------------------------------------------------------------
[net,info] = cnn_customtrain(net, imdb, fn, opts.train, 'conserveMemory', true, 'figure_interval', ceil(opts.train.numEpochs/8)) ; % ceil(1/opts.partial_traintest(1))


% -------------------------------------------------------------------------
% Testing
% -------------------------------------------------------------------------

% WARNING: in testing, no jittering, etc (no data augmentation)
if isfield(net.normalization, 'rgbVariance'), net.normalization = rmfield(net.normalization, 'rgbVariance'); end

% get a new imdb to not mess with the training one
% the new imdb will be used for computing AP
imdb_.name = 'fullNewTaskev';
imdb_ = newtaskopts.IMDBAddNewtask( imdb_, p, 3, 'trainval', [2], 'label', newtaskopts.imdblabelmode );

% for all old tasks
for i=1:accum_n_oldtasks
    % evaluate on the task
    if i > 1, partial_label = opts.accumu_batches{i-1}; else partial_label = []; end
    info_ = evalNet( oldtaskopts(i).evalmode, net, 2*i-1, 'partial_label', partial_label, 'getLastFCOutput', (i>1) );
    info.valall_OrigTask(i).val = info_.val;
    % compute Average Precision
    % assert task 1 never needs average precision
    if i > 1
        assert(isequal(oldtaskopts(i).IMDBAddOldtask, newtaskopts.IMDBAddNewtask));
        imdb__ = trimIMDBforBatch(imdb_, 3, partial_label);
        eval_labels = cat(2, imdb__.images.label{:});
        if size(eval_labels,1) == 1, eval_labels = dimensionalizeLabel(eval_labels, numel(partial_label)); end
        [info.valall_OrigTask(i).ap, info.valall_OrigTask(i).rec, info.valall_OrigTask(i).prec, info.valall_OrigTask(i).thresh] = ...
        VOCAveragePrecision( ...
            info.valall_OrigTask(i).val.lastfc_out{1}, ... % 1 == itask
            eval_labels, ...
            true); % (predictions, labels, perclass);
        fprintf('%s mean AP:\n', oldtaskopts(i).name);
        fprintf('%.3f \t', info.valall_OrigTask(i).ap'); fprintf('\n');
    end
end

% get the labels for the new task (for computing AP)
eval_labels = cell2mat(imdb_.images.label(imdb_.images.set == 2 & imdb_.images.task == 3)');
if accum_mode
    partial_label = opts.accumu_batches{accum_newtask_batch};
    if size(eval_labels,1) == 1
        invmap = zeros(1, max(eval_labels))+nan; invmap(partial_label) = 1:numel(partial_label); 
        eval_labels = invmap(eval_labels); eval_labels = eval_labels(~isnan(eval_labels));
    else
        eval_labels = eval_labels(partial_label, :);
    end
else
    partial_label = [];
end
% evaluate new task
[ info.valall_NewTask ] = evalNet( newtaskopts.evalmode, net, accum_n_oldtasks*2+1, 'partial_label', partial_label, 'getLastFCOutput', true );

% get the AP for the new task
% *: newtaskopts.newtaskdim changes if accumulative
if size(eval_labels,1) == 1, eval_labels = dimensionalizeLabel(eval_labels, newtaskopts.newtaskdim); end
[info.valall_NewTask.ap, info.valall_NewTask.rec, info.valall_NewTask.prec, info.valall_NewTask.thresh] = ...
    VOCAveragePrecision( ...
    info.valall_NewTask.val.lastfc_out{1}, ... % 1 == itask
    eval_labels, ...
    true); % (predictions, labels, perclass);
fprintf('%s AP:\n', newtaskopts.name);
fprintf('%.3f \t', info.valall_NewTask.ap'); fprintf('\n');
fprintf('%s mean AP:\n', newtaskopts.name);
disp(mean(info.valall_NewTask.ap));
save(fullfile(p.path_dump, 'info.mat'), 'info');


