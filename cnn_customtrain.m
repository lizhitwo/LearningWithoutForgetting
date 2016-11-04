function [net, info] = cnn_customtrain(net, imdb, getBatch, varargin)
% CNN_CUSTOMTRAIN   Training a CNN, customized for LwF.
%   CNN_CUSTOMTRAIN() is adapted from CNN_TRAIN() function from MatConvNet.
%   Major customizations include:
%     Tasks: allowing for different tasks to be run on a tree-structured 
%       Network. Different datasets for different tasks are specified by 
%       providing different GETBATCH functiion handles. Also see Paths.

%     Paths: allowing for running backprop on a tree-structured network.
%       We form sub-networks by selecting different root (input) to leaf (loss) 
%       paths. We perform parameter update on different paths alternatively 
%       between SGD steps.
%       The node selection for each sub-network is stored in NET.PATHS cell.
%       Note the difference of "task" in the paper and code. Code "task/paths"
%       may correspond to different loss forms of the same paper "task".

%       e.g. if the network has 2 tasks, 3 sub-networks may be used, #1 for 
%       training on the old task, #2 for keeping response on the old task 
%       (which has a different loss layer), and #3 for training on the new 
%       task. Joint training uses paths (1,3), LwF uses paths (2,3), and
%       Fine-tuning only uses path 3. These will be specified by arg TASKPATHS.

%       NET.PATHS will have three vectors specifying the layers involved with 
%       these 3 paths, and the TASKPATHS argument specifies the tasks out of 
%       the three that is being trained during this function call.

%     Keeping parameters using L2: a side experiment where we keep the network
%       parameters using a L2 loss instead of using LwF on the output.
% 
% 
% Authors: Zhizhong Li
% 
% See the COPYING file.

% Adapted from MatConvNet of VLFeat library. Their copyright info:

% Copyright (C) 2014-15 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

opts.batchSize = 256 ;
opts.numSubBatches = 1 ;
opts.train = [] ;
opts.val = [] ;
% TASKPATHS: IDs of the paths that are being trained. e.g. for LwF, this is usually [1,3]. For a simple_nn, this is [1].
if isfield(net, 'paths'), opts.taskPaths = 1:numel(net.paths); else opts.taskPaths = [1]; end % *NEW*
% MAJORTASK: ID of the main task that we use for deciding how long one epoch is; the other tasks are re-sampled to the same length.
opts.majorTask = 1;
% SHUFFLEFUNC: Function handle specifying how the data are shuffled each epoch.
%   This overrides the following two if corresponding function handle present
opts.shuffleFunc = cell(2,1);
% PARTIAL_DATA: Run shorter epochs. This is not subsampling the dataset; that
%   is done in IMDBAdd* functions.
%   e.g. [0.5,0.3] means run 0.5 training epoch and 0.3 test epoch and 
%   regard that as a full epoch. We treat [0,0] the same as [1,1].
opts.partial_data = [0 0];
% SHUFFLESYNCTASKS: making sure that when using the same dataset
opts.shuffleSyncTasks = [];
% SHUFFLEVAL: whether we use shuffled validation set each time. Note: if set to
%   false and partial_data(2) is used, the same subset will be selected each
%   epoch and result in biased validation results.
opts.shuffleVal = false; % *NEW*
opts.numEpochs = 300 ;
opts.gpus = [] ; % which GPU devices to use (none, one, or more)
opts.learningRate = 0.001 ;
% CONTINUE: whether to continue from saved network / optimizer state or ignore them.
opts.continue = false ;
opts.expDir = fullfile('data','exp') ;
opts.conserveMemory = true ;
opts.backPropDepth = +inf ;
% L2KEEPWEIGHTSDEPTH: if provided, regularize parameters to be similar to before 
%   using an L2 loss. 
%   ~DEPTH is the layers up to which the loss is applied.
%   ~DECAY is the weight of this loss within the total loss.
opts.l2keepweightsDepth = 0;
opts.l2keepweightsDecay = 0;
opts.sync = true ;
opts.prefetch = false ;
opts.cudnn = true ;
opts.weightDecay = 0.0005 ;
opts.momentum = 0.9 ;
opts.updateMethod = 'sgd-momentum';
% MOMENTUM2: a trial using Adam instead of sgd w/ momentum. Does not seem to 
%   provide benefit since we are not training from scratch and learning rate 
%   has to be small.
opts.momentum2 = 0.999 ;
opts.momentum2eps = 1e-8;
% ERRORFUNCTION: this can be different for each path. Use a string, or a handle, or a struct (see implementation)
opts.errorFunction = 'multiclass' ;
opts.errorLabels = {} ;
opts.plotDiagnostics = false ;
opts.memoryMapFile = fullfile(tempdir, 'matconvnet.bin') ;
% BALANCELOSS: the lambda for the loss for each task. For weighting the 
%   importance of each.
opts.balanceLoss = ones(numel(getBatch),1);
% GETLASTFCOUTPUT: if true, put the penultimate layer response into info.val.lastfc_out
opts.getLastFCOutput = true;
% FIGURE_INTERVAL: specify how often the loss/error plot is refreshed. 0 for silence.
opts.figure_interval = 1;
% NETATTACHLABELS: function handle. If specified, this will be used when 
%   attaching labels to the network instead of the default way.
opts.netAttachLabels = [];
% NETATTACHX: function handle. If specified, this will be used when attaching 
%   input to the network instead of the default way.
opts.netAttachX = [];
% FAKESAVE: if true, then instead of saving the training progress, a dummy save file is created to save space.
opts.fakesave = false;
opts = vl_argparse(opts, varargin) ;


% make directory for dumping results
if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end

% -------------------------------------------------------------------------
% Dealing with multiple tasks to jointly optimize them
% -------------------------------------------------------------------------
multiTaskMode = isa(getBatch, 'cell');
n_tasks = 1;
if multiTaskMode, n_tasks = max(imdb.images.task); end % # of tasks (or rather, paths) decided by imdb.
assert(opts.majorTask <= n_tasks);

% find the instance ID's inside imdb that belongs to the train/val set of
% each task.
if multiTaskMode
    if isempty(opts.train)
        alltrain = find(imdb.images.set==1) ;
        tasktrain = imdb.images.task(alltrain);
        opts.train = cell(n_tasks, 1);
        for i=1:numel(opts.train)
            opts.train{i} = alltrain(tasktrain == i);
        end
        clear alltrain tasktrain
    end
    if isempty(opts.val)
        allval = find(imdb.images.set==2) ;
        taskval = imdb.images.task(allval);
        opts.val = cell(n_tasks, 1);
        for i=1:numel(opts.val)
            opts.val{i} = allval(taskval == i);
        end
        clear allval taskval
    end
else
    if isempty(opts.train), opts.train = {find(imdb.images.set==1)} ; end
    if isempty(opts.val), opts.val = {find(imdb.images.set==2)} ; end
    getBatch = {getBatch};
end
if ~isa(opts.train, 'cell')
    if isnan(opts.train), opts.train = {[]} ; end
end

% specify back propagation depth for each epoch to reduce impact on old parameters. (only used for network expansion)
assert(numel(opts.backPropDepth)==1 || numel(opts.backPropDepth)==opts.numEpochs);

% -------------------------------------------------------------------------
% Network initialization
% -------------------------------------------------------------------------

% Decide whether this function call is evaluation only (specified by assigning arg TRAIN = NaN initially)
evaluateMode = all(cellfun( @isempty, opts.train(:) )) ;

if ~evaluateMode
  for i=1:numel(net.layers)
    % for each conv / fc layer... 
    if isfield(net.layers{i}, 'weights')
      J = numel(net.layers{i}.weights) ;
      % initialize SGD
      for j=1:J
        net.layers{i}.momentum{j} = zeros(size(net.layers{i}.weights{j}), 'single') ;
        if strcmp(opts.updateMethod, 'adam')
          net.layers{i}.momentum2{j} = zeros(size(net.layers{i}.weights{j}), 'single') ;
          net.layers{i}.updateCount = 0;
        end
      end
      if ~isfield(net.layers{i}, 'learningRate')
        net.layers{i}.learningRate = ones(1, J, 'single') ;
      end
      if ~iscell(net.layers{i}.learningRate)
        net.layers{i}.learningRate = num2cell(net.layers{i}.learningRate) ;
      end
      if ~isfield(net.layers{i}, 'weightDecay')
        net.layers{i}.weightDecay = ones(1, J, 'single') ;
      end

      % record original weights when L2KEEPWEIGHTSDEPTH is used.
      i_layer = min(cell2mat( cellfun(@(X) find(X==i), net.paths, 'UniformOutput', false) )); % find out what's the depth of this layer in any branches
      if ~isempty(i_layer) && i_layer <= opts.l2keepweightsDepth
        net.layers{i}.origweights = net.layers{i}.weights;
      else
        if isfield(net.layers{i}, 'origweights'), net.layers{i} = rmfield(net.layers{i}, 'origweights'); end
      end
    end

    % Legacy code: will be removed by MatConvNet
    if isfield(net.layers{i}, 'filters')
      assert(false);
      net.layers{i}.momentum{1} = zeros(size(net.layers{i}.filters), 'single') ;
      net.layers{i}.momentum{2} = zeros(size(net.layers{i}.biases), 'single') ;
      if strcmp(opts.updateMethod, 'adam')
        net.layers{i}.momentum2{1} = zeros(size(net.layers{i}.filters), 'single') ;
        net.layers{i}.momentum2{2} = zeros(size(net.layers{i}.biases), 'single') ;
      end
      if ~isfield(net.layers{i}, 'learningRate')
        net.layers{i}.learningRate = ones(1, 2, 'single') ;
      end
      if ~iscell(net.layers{i}.learningRate)
        net.layers{i}.learningRate = num2cell(net.layers{i}.learningRate) ;
      end
      if ~isfield(net.layers{i}, 'weightDecay')
        net.layers{i}.weightDecay = single([1 0]) ;
      end
    end
  end
end

% -------------------------------------------------------------------------
% Setting up GPU, evaluation, etc
% -------------------------------------------------------------------------

% setup GPUs
numGpus = numel(opts.gpus) ;
if numGpus > 1
  if isempty(gcp('nocreate')),
    parpool('local',numGpus) ;
    spmd, gpuDevice(opts.gpus(labindex)), end
  end
elseif numGpus == 1
  gpuDevice(opts.gpus)
end
if exist(opts.memoryMapFile), delete(opts.memoryMapFile) ; end

% setup error labels for train/val (string for the plots)
if all(cellfun(@isstr, opts.errorLabels))
  opts_errorLabels = opts.errorLabels;
  opts.errorLabels = cell(1,2);
  opts.errorLabels{1} = opts_errorLabels;
  opts.errorLabels{2} = opts_errorLabels;
else
  assert(all(cellfun(@iscell, opts.errorLabels)));
end
errorLabels.train = opts.errorLabels{1};
errorLabels.val = opts.errorLabels{2};
% after this, errorLabels.train/val will be a cell of error labels, each item (pair) for one task (that has evaluation).

% setup error calculation function
if ~iscell(opts.errorFunction)
  opts.errorFunction = repmat({opts.errorFunction}, [2, n_tasks]);
end
if numel(opts.errorFunction) == n_tasks
  opts.errorFunction = repmat(reshape(opts.errorFunction, 1,[]), [2, 1]);
end
% after this, opts.errorFunction will be a cell of 2*n_tasks, for train/val of each path.

% converting errorFunction strings to function handles
for i=1:numel(opts.errorFunction)
  if ischar(opts.errorFunction{i})
    switch opts.errorFunction{i}
      case 'none'
        opts.errorFunction{i} = @error_none ;
      case 'multiclass'
        opts.errorFunction{i} = @error_multiclass ;
      case 'multilabel' % for VOC where multiple objects may co-occur
        opts.errorFunction{i} = @error_multilabel ;

      % When using data augmentationfor validation/test set, the error functions are different.
      case 'multiclass_aug_f5' % ul, ur, bl, br, center
        opts.errorFunction{i} = @(opts, labels, res_pred_layer) error_multiclass(opts, labels, res_pred_layer, 10) ;
      case 'multilabel_aug_f5'
        opts.errorFunction{i} = @(opts, labels, res_pred_layer) error_multilabel(opts, labels, res_pred_layer, 10) ;
      case 'multiclass_aug_f6' % ul, ur, bl, br, center, whole
        opts.errorFunction{i} = @(opts, labels, res_pred_layer) error_multiclass(opts, labels, res_pred_layer, 12) ;
      case 'multilabel_aug_f6'
        opts.errorFunction{i} = @(opts, labels, res_pred_layer) error_multilabel(opts, labels, res_pred_layer, 12) ;
      case 'binary'
        opts.errorFunction{i} = @error_binary ;
      otherwise
        error('Uknown error function ''%s''', opts.errorFunction{i}) ;
    end
  end
end

% -------------------------------------------------------------------------
% Train and validate
% -------------------------------------------------------------------------

for epoch=1:opts.numEpochs
  info.val.lastfc_out = [];
  learningRate = opts.learningRate(min(epoch, numel(opts.learningRate))) ;

  % fast-forward to last checkpoint
  modelPath = @(ep) fullfile(opts.expDir, sprintf('net-epoch-%d.mat', ep));
  modelFigPath = fullfile(opts.expDir, 'net-train.pdf') ;

  % load network from files, but only overwrite the parameters / states, since 
  % the loss etc. may have changed from borrowing results from another experiment
  if opts.continue
    if exist(modelPath(epoch),'file')
      if epoch == opts.numEpochs
        [net, info] = overwriteNetExceptLossAndLock(net, info, modelPath(epoch));
      end
      continue ;
    end
    if epoch > 1
      fprintf('resuming by loading %s\n', modelPath(epoch-1)) ;
      [net, info] = overwriteNetExceptLossAndLock(net, info, modelPath(epoch-1));
    end
  end


  % time keeping
  tic_epoch = tic;

  % Announce the random seed for bookkeeping; use fixed value for debugging
  seed_thisepoch = randi(intmax()); 
  fprintf('Seed for epoch %d:\t%d\n', epoch, seed_thisepoch);
  rng(seed_thisepoch);
  parallel.gpu.rng(seed_thisepoch);

  % shuffling the training data. Some path's shuffling should be the same as 
  % the major path (e.g. LwF using the same images each batch for both tasks)

  % training data shuffling
  if isa(opts.shuffleFunc{1}, 'function_handle')
    [train] = opts.shuffleFunc{1}(opts.train, opts.partial_data(1), imdb);
  else
    train = cell(numel(opts.train), 1);
    % for shortening the epoch
    partial = opts.partial_data(1); if partial==0, partial = 1; end;
    % shuffle major task
    major_shuffle = randperm(numel(opts.train{opts.majorTask}), ceil(partial * numel(opts.train{opts.majorTask})));
    for i = 1:numel(opts.train)
      if (ismember(i, opts.shuffleSyncTasks) || i==opts.majorTask)
        % non-major task that is in sync with major task (has to use the same dataset)
        assert(all(size(opts.train{i}) == size(opts.train{opts.majorTask})));
        train{i} = opts.train{i}(major_shuffle);
      else
        % non-major task that is free
        train{i} = opts.train{i}(randperm(numel(opts.train{i}), ceil(partial * numel(opts.train{i})))) ;
      end
    end
  end

  % validation data shuffling
  if isa(opts.shuffleFunc{2}, 'function_handle')
    [val] = opts.shuffleFunc{2}(opts.val, opts.partial_data(2), imdb);
  else
    % for shortening the epoch
    partial = opts.partial_data(2); if partial==0, partial = 1; end;
    if opts.shuffleVal
      % actual shuffle
      for i = 1:numel(opts.val)
        val{i} = opts.val{i}(randperm(numel(opts.val{i}), ceil(partial * numel(opts.val{i})))) ; % shuffle
      end
    else
      % fake shuffle by sampling every (1/partial) sample.
      for i = 1:numel(opts.val)
        val{i} = opts.val{i}(floor(1:(1/partial):end)) ; % shuffle
      end
    end
  end
  

  % train one epoch, update network parameters, and aggregate results
  if numGpus <= 1
    [net,stats.train] = process_epoch(opts, getBatch, epoch, train, 1, learningRate, imdb, net) ;
    if opts.getLastFCOutput
      [~,stats.val,fcout_val] = process_epoch(opts, getBatch, epoch, val, 2, 0, imdb, net) ;
    else
      [~,stats.val] = process_epoch(opts, getBatch, epoch, val, 2, 0, imdb, net) ;
    end
  else
    warning('This part of multi-GPU code is not tested');
    spmd(numGpus)
      [net_, stats_train_] = process_epoch(opts, getBatch, epoch, train, 1, learningRate, imdb, net) ;
      if opts.getLastFCOutput
        [~, stats_val_, fcout_val_] = process_epoch(opts, getBatch, epoch, val, 2, 0, imdb, net_) ;
      else
        [~, stats_val_] = process_epoch(opts, getBatch, epoch, val, 2, 0, imdb, net_) ;
      end
    end
    net = net_{1} ;
    stats.train = sum([stats_train_{:}],2) ;
    stats.val = sum([stats_val_{:}],2) ;
    if opts.getLastFCOutput
      fcout_val = sum([fcout_val_{:}],2) ;
    end
  end
  
  if opts.getLastFCOutput % collect penultimate layer response
      info.val.lastfc_out = fcout_val;
  end

  % -------------------------------------------------------------------------
  % Preparing numbers and texts to plot
  % -------------------------------------------------------------------------
  % plot only validation or both?
  if evaluateMode, sets = {'val'} ; else sets = {'train', 'val'} ; end
  for f = sets
    f = char(f) ;
    % speed should be evaluated on major task's # sample
    n = numel(eval([f, '{opts.majorTask}'])) ; % each task' num_eval is the majorTask's sample's numel
    info.(f).speed(epoch) = n / stats.(f)(1) * max(1, numGpus) ;

    % convert all 'objective' (loss) labels to have a prefix of 'o:'
    % see if the errorLabels is just labels of errors, or objective+error combined
    if numel(errorLabels.(f))+2 == numel(stats.(f))
      errorLabels.(f) = [ ['o:' f], errorLabels.(f) ];
    end
    assert(numel(errorLabels.(f))+1 == numel(stats.(f)), 'Number of error names mismatches with error output by algorithm'); % stats.(f)'s first item is speed
    % which items are 'objectives' (losses) instead of errors?
    plotobj.(f) = strncmpi(errorLabels.(f), 'o:', 2); 

    if isfield(info.(f), 'errorLabels') && ~isequal(info.(f).errorLabels, errorLabels.(f))
      % this INFO is inherited from another experiment. We map it to a new INFO struct.
      disp('WARNING: the item names for objective/error have changed. Remapping. Error prone.');
      ind_origobj = strncmpi(info.(f).errorLabels, 'o:', 2);
      ind_currobj = plotobj.(f);

      str_origobj = info.(f).errorLabels(ind_origobj);
      str_currobj = errorLabels.(f)(ind_currobj);
      origobj = info.(f).objective;
      info.(f).objective(:) = 0;
      for iobj = 1:numel(str_currobj)
        iorigobj = find(strcmp(str_currobj{iobj}, str_origobj));
        if iorigobj
          info.(f).objective(iobj,:) = origobj(iorigobj,:);
        end
      end

      str_origerr = info.(f).errorLabels(~ind_origobj);
      str_currerr = errorLabels.(f)(~ind_currobj);
      origerr = info.(f).error;
      info.(f).error(:) = 0;
      for ierr = 1:numel(str_currerr)
        iorigerr = find(strcmp(str_currerr{ierr}, str_origerr));
        if iorigerr
          info.(f).error(ierr,:) = origerr(iorigerr,:);
        end
      end
    end
    % append results of all tasks to INFO.
    info.(f).objective(:,epoch) = stats.(f)(find(plotobj.(f))+1) / n ;
    info.(f).error(:,epoch) = stats.(f)(find(~plotobj.(f))+1) / n ;
    info.(f).errorLabels = errorLabels.(f);
  end

  % -------------------------------------------------------------------------
  % Saving the network training state.
  % -------------------------------------------------------------------------
  if ~evaluateMode
    if epoch >= opts.numEpochs
      % last epoch
      save(modelPath(epoch), 'net', 'info', '-v7.3') ;
    else
      % not last epoch; intermediate
      if isnumeric(opts.fakesave) || islogical(opts.fakesave) % opts.fakesave is 0/1/2
        if opts.fakesave % == 1 or 2
            if opts.fakesave == 1
              save(modelPath(epoch), 'epoch'); % 1: save dummy
            % 2: do nothing
            end
        else % 0: save everything
          save(modelPath(epoch), 'net', 'info', '-v7.3') ; 
        end
      elseif isstr(opts.fakesave) % opts.fakesave is string
        switch opts.fakesave
        case '@lrchange'
          learningRate_next = opts.learningRate(min(epoch+1, numel(opts.learningRate)));
          if abs(log(learningRate)-log(learningRate_next)) > log(2) % changed learning rate (difference>1/2)
            save(modelPath(epoch), 'net', 'info', '-v7.3') ; % save everything
          else
            save(modelPath(epoch), 'epoch'); % save dummy
          end
        end % switch opts.fakesave
      end % if opts.fakesave isnumeric/isstr
    end % if last epoch
  end % if just eval

  % -------------------------------------------------------------------------
  % Plotting figures.
  % -------------------------------------------------------------------------
  if opts.figure_interval > 0 && (mod(epoch,opts.figure_interval)==0 || epoch == opts.numEpochs)
    figure(1) ; clf ;
    hasError = all(all(cellfun(@(x) ~isempty(x), opts.errorFunction))) ;
    subplot(1,1+hasError,1) ;
    if ~evaluateMode
      semilogy(1:epoch, info.train.objective(:,1:epoch)', '.-', 'linewidth', 2) ;
      hold on ;
    end
    semilogy(1:epoch, info.val.objective(:,1:epoch)', '.--') ;
    xlabel('training epoch') ; ylabel('energy') ;
    grid on ;
    leg = {};
    for f = sets, f = char(f); leg = horzcat(leg, strcat(f, {' '}, errorLabels.(f)(plotobj.(f)))); end
    h=legend(leg) ;% what are the actual legends to be used?!
    set(h,'color','none');
    title('objective') ;
    if hasError
      subplot(1,2,2) ; leg = {} ;
      if ~evaluateMode
        plot(1:epoch, info.train.error(:,1:epoch)', '.-', 'linewidth', 2) ;
        hold on ;
        leg = horzcat(leg, strcat({'train '}, errorLabels.train(~plotobj.train))) ;
      end
      plot(1:epoch, info.val.error(:,1:epoch)', '.--') ;
      leg = horzcat(leg, strcat({'val '}, errorLabels.val(~plotobj.val))) ;
      set(legend(leg{:}),'color','none') ;
      grid on ;
      xlabel('training epoch') ; ylabel('error') ;
      title('error') ;
    end
    drawnow ;
    print(1, modelFigPath, '-dpdf') ;
  end
  toc(tic_epoch);
end


% -------------------------------------------------------------------------
function [predictions, labels] = prediction_augment(predictions, labels, n_augment)
% using the mean of "the prediction scores of all the augmented versions of the same image" to get "augmented prediction score".
sizepred = size(predictions);
assert(mod(sizepred(4),n_augment)==0);
predictions = reshape(predictions, sizepred(1), sizepred(2), sizepred(3), n_augment, sizepred(4)/n_augment);
predictions = reshape(mean(predictions, 4), sizepred(1), sizepred(2), sizepred(3), sizepred(4)/n_augment);
assert(isvector(labels));
labels = labels(1:n_augment:end);

% -------------------------------------------------------------------------
function [net, info] = overwriteNetExceptLossAndLock(net, info, netfile)
%% overwrite net with everything in loadnet.net except the last layer of each path
% this is done by copying NET to loaded net and returning loaded net.
loadnet = load(netfile, 'net', 'info') ;

if isfield(loadnet, 'net')
  assert(isfield(net, 'paths') == isfield(loadnet.net, 'paths'));
  if isfield(net, 'paths')
    % check for consistency between NET and the loaded one
    assert(all(cellfun(@numel, net.paths)==cellfun(@numel, loadnet.net.paths)));
    if ~isequal(loadnet.net.paths, net.paths)
      % map layers to layers
      orig_layerind = cat(2,loadnet.net.paths{:});
      this_layerind = cat(2,net.paths{:});
      map = zeros(1,numel(loadnet.net.layers)); map(orig_layerind) = this_layerind;
      newloadpath = cellfun(@(X) map(X), loadnet.net.paths, 'UniformOutput', false);
      assert(isequal(newloadpath, net.paths));
      loadnet.net.layers(map) = loadnet.net.layers;
      loadnet.net.paths = newloadpath;
    end
    for ipath = 1:numel(net.paths)
      % make sure layers' types are the same
      type2 = cellfun(@(X) X.type, net.layers(net.paths{ipath}), 'UniformOutput', false);
      type1 = cellfun(@(X) X.type, loadnet.net.layers(loadnet.net.paths{ipath}), 'UniformOutput', false);
      assert(isequal(type1, type2));
      % make sure layers' names are the same
      hasname2 = cellfun(@(X) isfield(X, 'name'), net.layers(net.paths{ipath}));
      hasname1 = cellfun(@(X) isfield(X, 'name'), loadnet.net.layers(loadnet.net.paths{ipath}));
      name2 = cellfun(@(X) X.name, net.layers(net.paths{ipath}(hasname2)), 'UniformOutput', false);
      name1 = cellfun(@(X) X.name, loadnet.net.layers(loadnet.net.paths{ipath}(hasname1)), 'UniformOutput', false);
      assert(all(hasname1==hasname2) && isequal(name1, name2));

      % Finally, copy over new loss layer
      loadnet.net.layers{net.paths{ipath}(end)} = net.layers{net.paths{ipath}(end)};
    end
  else
    loadnet.net.layers{end} = net.layers{end};
  end

  % use NET's learning rate and origweights
  for i=1:numel(net.layers)
    if isfield(net.layers{i}, 'learningRate')
      loadnet.net.layers{i}.learningRate = net.layers{i}.learningRate;
    else
      assert(~isfield(net.layers{i}, 'weights'), 'How can a layer not have a .learningRate? It should have been written into it if there is none!');
      if isfield(loadnet.net.layers{i}, 'learningRate')
        loadnet.net.layers{i} = rmfield(loadnet.net.layers{i}, 'learningRate');
      end
    end
    if isfield(net.layers{i}, 'origweights'), loadnet.net.layers{i}.origweights = net.layers{i}.origweights;
    else
      if isfield(loadnet.net.layers{i}, 'origweights'), loadnet.net.layers{i} = rmfield(loadnet.net.layers{i}, 'origweights'); end
    end
  end
  net = loadnet.net;
end

% overwrite info altogether
if isfield(loadnet, 'info')
  info = loadnet.info;
end

% -------------------------------------------------------------------------
function err = error_multiclass(opts, labels, res_pred_layer, n_augment)
% -------------------------------------------------------------------------
if nargin < 4, n_augment = 1; end
predictions = gather(res_pred_layer.x) ;
if n_augment > 1, [ predictions, labels ] = prediction_augment(predictions, labels, n_augment); end
[~,predictions] = sort(predictions, 3, 'descend') ;

% be resilient to badly formatted labels
if numel(labels) == size(predictions, 4)
  labels = reshape(labels,1,1,1,[]) ;
end
assert(size(labels, 4) == size(predictions, 4), 'WARNING: error cannot be computed using labels');

% skip null labels
mass = single(labels(:,:,1,:) > 0) ;
if size(labels,3) == 2
  % if there is a second channel in labels, used it as weights
  mass = mass .* labels(:,:,2,:) ;
  labels(:,:,2,:) = [] ;
end
error = ~bsxfun(@eq, predictions, labels) ;
err(1,1) = sum(sum(sum(mass .* error(:,:,1,:)))) ;
top5_actual = min(5,size(error,3));
err(2,1) = sum(sum(sum(mass .* min(error(:,:,1:top5_actual,:),[],3)))) ;

% -------------------------------------------------------------------------
function err = error_multilabel(opts, labels, res_pred_layer, n_augment)
% -------------------------------------------------------------------------
if nargin < 4, n_augment = 1; end
predictions = gather(res_pred_layer.x) ;
if n_augment > 1, [ predictions, labels ] = prediction_augment(predictions, labels, n_augment); end

% be resilient to badly formatted labels
if ndims(labels) ~= 4
  assert(ndims(labels) == 2);
  labels = reshape(labels,[1,1,size(labels)]) ;
end

assert(all(size(labels) == size(predictions)), 'Label size error: mismatch with predictions');
% no null labels allowed
error = (predictions > 0) ~= labels ;
err(1,1) = sum(sum(sum(mean(error,3)))) ;
% a dumbly mimicked top5 error: # error classes >= 5
err(2,1) = sum(sum(sum(sum(error,3) >= 5))) ;
err = single(err);

% -------------------------------------------------------------------------
function err = error_multilabel_avgpr(opts, labels, res_pred_layer)
% -------------------------------------------------------------------------
% note that averaging this over batches makes no sense.
predictions = gather(res_pred_layer.x) ;
predictions = reshape(predictions,[size(predictions,3),size(predictions,4)]) ;

% be resilient to badly formatted labels
if ndims(labels) ~= 4
  assert(ndims(labels) == 2);
else
  labels = reshape(labels,[size(labels,3),size(labels,4)]) ;
end

assert(all(size(labels) == size(predictions)), 'Label size error: mismatch with predictions');

% use VOC's code for Average Precision
ap = VOCAveragePrecision(predictions, labels, false);

% no null labels allowed
error = (predictions > 0) ~= labels ;
err(1,1) = ap*size(labels,2);
% a dumbly mimicked top5 error: # error classes >= 5
err(2,1) = sum(mean(error,1)) ;
err = single(err);


% -------------------------------------------------------------------------
function err = error_binary(opts, labels, res_pred_layer)
% -------------------------------------------------------------------------
predictions = gather(res_pred_layer.x) ;
error = bsxfun(@times, predictions, labels) < 0 ;
err = single(sum(error(:))) ;

% -------------------------------------------------------------------------
function err = error_none(opts, labels, res_pred_layer)
% -------------------------------------------------------------------------
err = zeros(0,1, 'single') ;

% -------------------------------------------------------------------------
function  [net_cpu,stats,last_fc_out,prof] = process_epoch(opts, getBatch, epoch, subset, trainval, learningRate, imdb, net_cpu)
% -------------------------------------------------------------------------

% validate the number of tasks
assert(isa(getBatch, 'cell') && isa(subset, 'cell'));
assert(numel(getBatch) == numel(subset));
assert(numel(getBatch) == numel(opts.taskPaths));
n_tasks = numel(subset);
assert(opts.majorTask <= n_tasks);

% move CNN to GPU as needed
numGpus = numel(opts.gpus) ;
if numGpus >= 1
  net = vl_customnn_move(net_cpu, 'gpu') ;
else
  net = net_cpu ;
  net_cpu = [] ;
end

% validation mode if learning rate is zero
training = learningRate > 0 ;
if training, mode = 'training' ; else, mode = 'validation' ; end
if nargout > 3, mpiprofile on ; end

numGpus = numel(opts.gpus) ;
if numGpus >= 1
  one = gpuArray(single(1)) ;
  dzdzi = gpuArray(single(opts.balanceLoss));
else
  one = single(1) ;
  dzdzi = single(opts.balanceLoss);
end
res = [] ;
mmap = [] ;
stats = [] ;
xlast_record = cell(n_tasks, 0);

% Now we count epoch on the majorTask; for the rest tasks, using the 
% same number of pictures for each task and basically ignore extra ones and
% loop where insufficient...

% loop over batch; within the loop, loop over paths
for t=1:opts.batchSize:numel(subset{opts.majorTask})
  fprintf('%s: epoch %02d: batch %3d/%3d: ', mode, epoch, ...
          fix(t/opts.batchSize)+1, ceil(numel(subset{opts.majorTask})/opts.batchSize)) ;
  batchSize = min(opts.batchSize, numel(subset{opts.majorTask}) - t + 1) ;
  batchTime = tic ;
  numDone = 0 ;
  error = repmat({zeros(0, 'single')},[n_tasks,1]);
  nextBatch = [];
  
  for itask = 1:n_tasks
      flag_resetres = true; %accumtask%
      if numel(subset{itask}) == 0, continue; end
      for s=1:opts.numSubBatches

        % get this image batch.
        batchStart = t + (labindex-1) + (s-1) * numlabs ;
        batchEnd = min(t+opts.batchSize-1, numel(subset{opts.majorTask})) ; % count numbers on the majorTask
        batch = subset{itask}( mod( (batchStart-1 : opts.numSubBatches * numlabs : batchEnd-1), numel(subset{itask}) )+1 ) ; % loop index for current task where necessary
        if ~isempty(nextBatch), assert(all(nextBatch==batch)); end
        [im, labels] = getBatch{itask}(imdb, batch) ;

        % pre-fetch next batch
        if opts.prefetch 
          % for pre-fetching, the following chunk of code determines what the next subbatch is
          if s==opts.numSubBatches
            % continuing on the next task's first subbatch... 
            ipreftask = itask;
            batchStart = t + (labindex-1) ;
            batchEnd = min(t+opts.batchSize-1, numel(subset{opts.majorTask})) ;
            while true
                ipreftask = ipreftask+1;
                % if next task loops over, then next batch's first task's ~
                if ipreftask > n_tasks
                    ipreftask = 1;
                    batchStart = batchStart + opts.batchSize;
                    batchEnd = min(batchEnd + opts.batchSize, numel(subset{opts.majorTask}));
                end
                if numel(subset{ipreftask}) ~= 0, break; end
            end
          else
            % continuing on this batch's this task's next sub-batch
            ipreftask = itask;
            batchStart = batchStart + numlabs ;
          end

          % now get the batch ID's and prefetch
          if batchStart <= numel(subset{opts.majorTask})
              nextBatch = subset{ipreftask}( mod( (batchStart-1 : opts.numSubBatches * numlabs : batchEnd-1), numel(subset{ipreftask}) )+1 ) ;
              getBatch{ipreftask}(imdb, nextBatch) ;
          end
        end

        if numGpus >= 1
          if iscell(im)
            im = cellfun(@gpuArray, im, 'UniformOutput', false);
          else
            im = gpuArray(im) ;
          end
          wait(gpuDevice);
        end

        % provide evaluation to compute loss
        if isa(opts.netAttachLabels, 'function_handle')
          % if a way to attach labels to net is provided
          net = opts.netAttachLabels(net, labels);
          assert(~isfield(net, 'paths'), 'Multi-path has not been implemented for customized label attaching');
          ipath = 1; % placeholder actually...
        else
          % figure out which path we are performing learning on, and which layer
          % is this path's last layer. Then attach the labels to the correct
          % last layer
          multiTaskMode = numel(getBatch) > 1;
          if multiTaskMode
            assert( isfield(net, 'paths') );
            ipath = opts.taskPaths(itask);
            net.layers{ net.paths{ipath}(end) }.class = labels ;
          else
            if isfield(net, 'paths')
              % according to assumptions, numel(opts.taskPath) == 1,
              % and opts.majorTask == 1, and itask == 1 too.
              ipath = opts.taskPaths(itask);
              net.layers{ net.paths{ipath}(end) }.class = labels ;
            else
              assert(opts.taskPaths==1 && opts.majorTask==1);
              ipath = 1; % placeholder actually...
              net.layers{end}.class = labels ;
            end
          end
        end

        if numGpus >= 1, wait(gpuDevice); end
        
        % actually running the convnet (get differential)
        if training, dzdy = dzdzi(itask); else, dzdy = [] ; end % each task has different lambdas. Use them
        res = vl_graphnn(net, im, dzdy, res, ...
                          'nn_path', ipath, ...
                          'netAttachX', opts.netAttachX, ...
                          'accumulate', ~flag_resetres, ... %accumtask%
                          'disableDropout', ~training, ...
                          'conserveMemory', opts.conserveMemory, ...
                          'backPropDepth', opts.backPropDepth(min(epoch, numel(opts.backPropDepth))), ...
                          'sync', opts.sync, ...
                          'cudnn', opts.cudnn) ;

        flag_resetres = false;
        % accumulate training errors
        if isfield(net, 'paths')
          last_layer = net.paths{ipath}(end); % layer# of the softmaxloss or whatever
        else
          last_layer = numel(net.layers); % layer# of the softmaxloss or whatever
        end

        % compute error using errorFunction -- either a handle, or a struct.
        if isstruct(opts.errorFunction{trainval, itask})
          fun = opts.errorFunction{trainval, itask};
          switch fun.arg
            case 'full_result'
              errorcall = fun.fun(opts, labels, res);
            otherwise
              errorcall = fun.fun(opts, labels, res(last_layer));
          end
        else
          errorcall = opts.errorFunction{trainval, itask}(opts, labels, res(last_layer));
        end
        
        % sum past errors with current batch errors (or losses)
        error{itask} = sum([error{itask}, [...
          sum(double(gather(res(end).x))) ;
          reshape(errorcall,[],1) ; ]],2) ;
        assert(isa(error{itask},'single'));

        % if requires penultimate layer response, record them
        if nargout > 2
          fcompact = @(x) reshape(double(gather(x)), [size(x,3) size(x,4)]);
          if iscell(res(last_layer).x)
              xlast_tmp = cellfun(fcompact, res(last_layer).x, 'UniformOutput', false);
              xlast_record{itask,end+1} = cat(1,xlast_tmp{:});
              clear xlast_tmp;
          else
              xlast_record{itask,end+1} = fcompact(res(last_layer).x);
          end
        end
        if itask == opts.majorTask, numDone = numDone + numel(batch) ; end
      end

      % gather and accumulate gradients across labs (<- the SGD is here)
      % SGD for each task.
      if training
        if numGpus <= 1
          [net,res] = accumulate_gradients(opts, learningRate, batchSize, net, res) ;
        else
          if isempty(mmap)
            mmap = map_gradients(opts.memoryMapFile, net, res, numGpus) ;
          end
          write_gradients(mmap, net, res) ;
          labBarrier() ;
          [net,res] = accumulate_gradients(opts, learningRate, batchSize, net, res, mmap) ;
        end
      end
  end % for itask %accumtask%

  % print learning statistics
  batchTime = toc(batchTime) ;
  stats = sum([stats,[ batchTime ; cell2mat(error) ]],2); % works even when stats=[]
  speed = batchSize/batchTime ;

  fprintf(' %.2f s (%.1f data/s)', batchTime, speed) ;
  n = (t + batchSize - 1) / max(1,numlabs) ;
  if numel(opts.errorLabels{trainval})+1 == numel(stats)
      stat_offset = 1;
  else
      assert(numel(opts.errorLabels{trainval})+2 == numel(stats));
      stat_offset = 2;
      fprintf(' obj:%.3g', stats(2)/n) ;
  end
  for i=1:numel(opts.errorLabels{trainval})
    fprintf(' %s:%.3g', opts.errorLabels{trainval}{i}, stats(i+stat_offset)/n) ;
  end
  fprintf(' [%d/%d]', numDone, batchSize);
  fprintf('\n') ;

  % debug info
  if opts.plotDiagnostics && numGpus <= 1
    figure(2) ; vl_simplenn_diagnose(net,res) ; drawnow ;
  end
end

% if requires penultimate layer response, report them
if nargout > 2
  last_fc_out = cell(n_tasks,1);
  for itask = 1:n_tasks
      last_fc_out{itask} = cat(2,xlast_record{itask,:});
  end
end

if nargout > 3
  prof = mpiprofile('info');
  mpiprofile off ;
end

if numGpus >= 1
  net_cpu = vl_customnn_move(net, 'cpu') ;
else
  net_cpu = net ;
end

% -------------------------------------------------------------------------
function [net,res] = accumulate_gradients(opts, lr, batchSize, net, res, mmap)
% -------------------------------------------------------------------------

for l=numel(net.layers):-1:1
  if numel(res(l).dzdw)
    if isfield(net.layers{l}, 'updateCount')
      net.layers{l}.updateCount = net.layers{l}.updateCount + 1;
    else
      net.layers{l}.updateCount = 1;
    end
  end
  for j=1:numel(res(l).dzdw)
    thisDecay = opts.weightDecay * net.layers{l}.weightDecay(j) ;
    thisLR = lr * net.layers{l}.learningRate{j} ;

    % the option to us a l2 loss to make parameters close to their original values.
    % here we decide the gradient from that loss.
    if opts.l2keepweightsDecay == 0 || ~isfield(net.layers{l}, 'origweights'), thisL2keepW = 0; 
    else thisL2keepW = opts.l2keepweightsDecay * (net.layers{l}.weights{j} - net.layers{l}.origweights{j}); 
    end

    % accumualte from multiple labs (GPUs) if needed
    if nargin >= 7
      tag = sprintf('l%d_%d',l,j) ;
      tmp = zeros(size(mmap.Data(labindex).(tag)), 'single') ;
      for g = setdiff(1:numel(mmap.Data), labindex)
        tmp = tmp + mmap.Data(g).(tag) ;
      end
      res(l).dzdw{j} = res(l).dzdw{j} + tmp ;
    end

    % descent
    if isfield(net.layers{l}, 'weights')
      switch opts.updateMethod
        case 'sgd-momentum'
          net.layers{l}.momentum{j} = ...
            opts.momentum * net.layers{l}.momentum{j} ...
            - thisDecay * net.layers{l}.weights{j} ...
            - thisL2keepW ...
            - (1 / batchSize) * res(l).dzdw{j} ;
          net.layers{l}.weights{j} = net.layers{l}.weights{j} + bsxfun(@times, thisLR, net.layers{l}.momentum{j}) ;
        case 'adam'
          dw = thisDecay * net.layers{l}.weights{j} + thisL2keepW + (1 / batchSize) * res(l).dzdw{j};
          net.layers{l}.momentum{j} = ...
            opts.momentum * net.layers{l}.momentum{j} ...
            + (1-opts.momentum) * dw ;
          net.layers{l}.momentum2{j} = ...
            opts.momentum2 * net.layers{l}.momentum2{j} ...
            + (1-opts.momentum2) * (dw .* dw) ;
          net.layers{l}.weights{j} = net.layers{l}.weights{j} ...
            - bsxfun(@times, thisLR / (1-opts.momentum^net.layers{l}.updateCount), net.layers{l}.momentum{j}) ...
              ./ (sqrt(net.layers{l}.momentum2{j} / (1-opts.momentum2^net.layers{l}.updateCount)) + opts.momentum2eps) ;
          clear dw
        otherwise
          assert(false, 'unrecognized updateMethod');
      end

      % check if there are NaN's or Inf's in the values. 
      if any(isnan(net.layers{l}.weights{j}(:)))
          keyboard;
          if isa(net.layers{l}.weights{j}, 'gpuArray')
            fprintf('Occurrence of nan in GPU weights but not CPU weights?????\n');
            net.layers{l}.weights{j} = gpuArray(gather(net.layers{l}.weights{j}));
            assert(~any(isnan(net.layers{l}.weights{j}(:))));
          else
            fprintf('\nOccurrence of nan in weights\n\n');
            assert(false);
          end
      end
      if any(isinf(net.layers{l}.weights{j}(:)))
          fprintf('\nOccurrence of +/- inf in weights\n\n');
          % keyboard;
          assert(false);
      end
    else
      % Legacy code: to be removed by MatConvNet
      assert(strcmp(opts.updateMethod, 'sgd-momentum'));
      if j == 1
        net.layers{l}.momentum{j} = ...
          opts.momentum * net.layers{l}.momentum{j} ...
          - thisDecay * net.layers{l}.filters ...
          - (1 / batchSize) * res(l).dzdw{j} ;
        net.layers{l}.filters = net.layers{l}.filters + bsxfun(@times, thisLR, net.layers{l}.momentum{j}) ;
      else
        net.layers{l}.momentum{j} = ...
          opts.momentum * net.layers{l}.momentum{j} ...
          - thisDecay * net.layers{l}.biases ...
          - (1 / batchSize) * res(l).dzdw{j} ;
        net.layers{l}.biases = net.layers{l}.biases + bsxfun(@times, thisLR, net.layers{l}.momentum{j}) ;
      end
    end
  end
end

% -------------------------------------------------------------------------
function mmap = map_gradients(fname, net, res, numGpus)
% -------------------------------------------------------------------------
format = {} ;
for i=1:numel(net.layers)
  for j=1:numel(res(i).dzdw)
    format(end+1,1:3) = {'single', size(res(i).dzdw{j}), sprintf('l%d_%d',i,j)} ;
  end
end
format(end+1,1:3) = {'double', [3 1], 'errors'} ;
if ~exist(fname) && (labindex == 1)
  f = fopen(fname,'wb') ;
  for g=1:numGpus
    for i=1:size(format,1)
      fwrite(f,zeros(format{i,2},format{i,1}),format{i,1}) ;
    end
  end
  fclose(f) ;
end
labBarrier() ;
mmap = memmapfile(fname, 'Format', format, 'Repeat', numGpus, 'Writable', true) ;

% -------------------------------------------------------------------------
function write_gradients(mmap, net, res)
% -------------------------------------------------------------------------
for i=1:numel(net.layers)
  for j=1:numel(res(i).dzdw)
    mmap.Data(labindex).(sprintf('l%d_%d',i,j)) = gather(res(i).dzdw{j}) ;
  end
end
