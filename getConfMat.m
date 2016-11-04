function [ cmat, cmatcount ] = getConfMat( label, pred, varargin )
% GETCONFMAT   generate the confusion matrix based on labels (LABEL) and prediction (PRED).
%   Output: 
%     CMAT confusion matrix in probability format.
%     CMATCOUNT confusion matrix in number count format.
%   Options:
%     See code comments
% 
% Authors: Zhizhong Li
% 
% See the COPYING file.

opts.figure = false; % whether to plot the confusion matrix.
opts.print = false; % whether to save the plot to a file (0: no; file string: yes)
opts.nclasses = max([label(:); pred(:)]); % specify the total number of classes. Otherwise use the largest in LABEL and PRED.
opts.className = []; % name of the classes when plotting the matrix.
opts = vl_argparse(opts, varargin);

nclasses = opts.nclasses;
cmatcount = accumarray([label(:), pred(:)], ones(numel(label), 1), [nclasses,nclasses]);
gtcount = sum(cmatcount,2);
predcount = diag(cmatcount);
accupercls = predcount./gtcount;
cmat = bsxfun(@rdivide, cmatcount, gtcount);

if opts.figure
    figure(double(opts.figure)); imagesc(cmat); set(gca,'dataAspectRatio',[1 1 1]); colormap('jet'); colorbar; caxis([0 1]);
    ncls = size(cmat,1);
    ax = gca; ax.XTick = 1:ncls; ax.YTick = 1:ncls; ax.XTickLabelRotation = 45;
    if isempty(opts.className)
        segclasses = load('ssegClass.mat'); segclasses = segclasses.cls; segclasses{1} = 'BadBox'; 
    else
        segclasses = opts.className;
    end
    ax.XTickLabel = segclasses; ax.YTickLabel = strcat(segclasses(:), arrayfun(@(x) sprintf('  ts%d / %2.1f%%', gtcount(x), accupercls(x)*100), (1:numel(gtcount))', 'UniformOutput', false));
    hold on;
    gridx = plot(repmat(ax.XLim', 1, ncls), [ax.YTick; ax.YTick]+.5, 'w-', 'Color', [1,1,1,0.2]); % alpha(gridx,.5);
    gridy = plot([ax.XTick; ax.XTick]+.5, repmat(ax.YLim', 1, ncls), 'w-', 'Color', [1,1,1,0.2]); % alpha(gridy,.5);
    hold off;
    if opts.print
        ax.FontSize = 6; if numel(segclasses) > 70, ax.FontSize = 3; end
        print(double(opts.figure), opts.print, '-dpdf') ;
    end
end
