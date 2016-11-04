function [imdb, isin] = trimIMDBforBatch(imdb, task, label_batch)
% TRIMIMDBFORBATCH   delete in the task TASK all samples with labels not in 
% LABEL_BATCH. Then, reformat the label as if the task dataset only has the 
% class entries in LABEL_BATCH. Works on both multiclass and multilabel styles.
%   Input:
%     MSGTXT the headline text
%     DETAILS the content text
% 
% Authors: Zhizhong Li
% 
% See the COPYING file.

% pick out samples for this task
taskentries = find(imdb.images.task==task);
labelmat = cat(2, imdb.images.label{taskentries});

% figure out which is used and transform label: multiclass or multilabel
if size(labelmat,1) == 1
    % multiclass style
    isin = logical(sum( bsxfun(@eq, labelmat, label_batch(:)), 1));
    invmap = zeros(1, max(label_batch))+nan; invmap(label_batch) = 1:numel(label_batch);
    labelmat = labelmat(isin);
    labelmat = invmap(labelmat);
    assert(~any(isnan(labelmat(:))));
else
    % multilabel style (VOC)
    labelmat = labelmat(label_batch,:);
    isin = sum(labelmat, 1);
    %VOCfull% For multilabel, we do not delete any samples if they do not have 
    % the LABEL_BATCH labels, but rather, keep them as negative samples.
    isin = true(size(isin));
end

% delete the rest
todel = taskentries(~isin);
tochange = taskentries(isin);

% change first, delete second; otherwise index changes
assert(all(imdb.images.task(tochange) == task));
imdb.images.label(tochange) = num2cell( labelmat, 1 )';

imdb.images.name(todel) = [];
imdb.images.label(todel) = [];
imdb.images.set(todel) = [];
assert(all(imdb.images.task(todel) == task));
imdb.images.task(todel) = [];

