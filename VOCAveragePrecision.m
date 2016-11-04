function [ ap, rec, prec, thresh ] = VOCAveragePrecision(predictions, labels, perclass, draw)
% VOCAVERAGEPRECISION average precision for multiple classes.
% 
% Input:
%   PREDICTIONS, LABELS confidence and 1/0 labels for each class (row for class) and each sample (column).
%   PERCLASS whether the AP for each class is returned (true) or the Mean AP is returned (false).
%   DRAW whether to plot the PR curve or not.
% 
% Return:
%   AP the average precision for each class, or the Mean AP.
%   REC, PREC, THRESH recall, precision, threshold for each sorted sample.
% 
% Authors: Zhizhong Li
% 
% See the COPYING file.
% 
% Adapted from VOC devkit code: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit

ap = zeros(size(labels,1),1);
n_cls = size(labels,1);
rec = zeros(size(labels,2),n_cls);
prec = zeros(size(labels,2),n_cls);
thresh = zeros(size(labels,2),n_cls);
for iclass = 1:size(labels,1)
  out = predictions(iclass,:);
  gt = labels(iclass,:);
  out = out(:); gt = gt(:);

  [so,si]=sort(-out);
  tp=gt(si)>0;
  fp=gt(si)<=0;

  fp=cumsum(fp);
  tp=cumsum(tp);
  rec(:,iclass)=tp/sum(gt>0);
  prec(:,iclass)=tp./(fp+tp);
  thresh(:,iclass) = -so;

  ap(iclass)=VOCap(rec(:,iclass),prec(:,iclass));

  if nargin >= 4 && draw
      % plot precision/recall
      plot(rec(:,iclass),prec(:,iclass),'-');
      grid;
      xlabel 'recall'
      ylabel 'precision'
      xlim([0 1]); ylim([0 1]); 
      title(sprintf('class: %d, AP = %.3f, #samples = %d', iclass, ap(iclass), sum(gt>0)));
      if draw >= 2, w = waitforbuttonpress; end
  end
end

if ~perclass, ap = mean(ap); end


function ap = VOCap(rec,prec)

mrec=[0 ; rec ; 1];
mpre=[0 ; prec ; 0];
for i=numel(mpre)-1:-1:1
    mpre(i)=max(mpre(i),mpre(i+1));
end
i=find(mrec(2:end)~=mrec(1:end-1))+1;
ap=sum((mrec(i)-mrec(i-1)).*mpre(i));

