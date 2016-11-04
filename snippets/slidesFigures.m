% perf = [56.5, 75.8, 55.1, 57.5, 55.9, 64.5, 43.3, 72.1, 38.4, 41.7, 43.0, 75.3, 52.1, 99.0   
%     -1.4, -0.3, -5.1, -1.5, -3.4, -1.0, -1.8, -0.1, -9.1, -0.8, -4.1, -0.8, -4.9, 0.2    
%     0.5, -1.1, 2.0, -5.3, 1.2, -3.7, -0.2, -3.9, 4.7, -19.4, 0.2, -0.5, 5.0, -0.8    
%     0.2, 0.0, 0.5, -0.9, 0.5, -0.6, -0.1, 0.1, 3.3, -0.2, 0.2, 0.1, 4.7, 0.2 ];

% fine-tuning
% joint-training
% feature-extractor
% learning-without-forgetting
% new old
function slidesFigures

dumpdir = 'temp/'

perf = [75.51386667,	55.12533333,	56.02923333,	49.97333333,	63.43283333,	52.50266667,	71.95046667,	41.50953333,	40.91016667,	29.31666667,	74.50246667,	38.91106667,	99.24,	47.136
    75.83703333,	56.72066667,	56.58726667,	55.54,	63.88056667,	56.38,	72.21593333,	43.20363333,	41.54873333,	41.71406667,	75.4229,	43.16543333,	99.18,	56.756
    74.77316667,	57.032,	52.21493333,	57.032,	60.8209,	57.032,	68.17626667,	43.1372,	22.33346667,	43.1372,	74.77613333,	43.1372,	98.21,	57.032
    75.83233333,	56.48666667,	57.48476667,	55.06666667,	64.4776,	55.86933333,	72.09943333,	43.29173333,	41.70983333,	38.44223333,	75.2985,	42.96926667,	99.03,	52.08]

titles = {'AlexImNet-VOC', 'AlexImNet-CUB', 'AlexImNet-scenes67', 'AlexPlaces2-VOC', 'AlexPlaces2-CUB', 'AlexPlaces2-scenes67', 'AlexImNet-MNIST'};
colors = {'r','g','c','b'};

mtd_order = [3 1 2];
tpair_order = [4 5 6 1 2 3 7];
mtd_str = {'ex', 'ft', 'jt'};
mtd_string = {'Feature Extraction', 'Fine-tuning', 'Joint Training'};
n_tpairs = size(perf,2)/2;

% ========== point plot ========== 
% mtd_order(4) = 4;
% perf_ = bsxfun(@minus, perf, perf(4,:));
% subplot(1,2,1); plot(repmat((1:4)',1,7), perf_(mtd_order,2:2:end), 'k:'); hold on;
% subplot(1,2,2); plot(repmat((1:4)',1,7), perf_(mtd_order,1:2:end), 'k:'); hold on;
% for i=1:4
%     subplot(1,2,1);
%     scatter(i*ones(1,7), perf_(mtd_order(i),2:2:end), 'MarkerFaceColor', colors{mtd_order(i)});
%     subplot(1,2,2);
%     scatter(i*ones(1,7), perf_(mtd_order(i),1:2:end), 'MarkerFaceColor', colors{mtd_order(i)});
% end
% subplot(1,2,1); xlim([0.5,4.5]); hold off; ax = gca; ax.XTickLabel = {}; box off;
% subplot(1,2,2); xlim([0.5,4.5]); hold off; ax = gca; ax.XTickLabel = {}; box off;


% ========== scatter plot ========== 
% for imtd=1:3
%     i = mtd_order(imtd);
%     figure(1);
%     clf('reset');
%     set(gcf, 'PaperPosition', [0.25 2.5 8 3]);

%     subplot(1,3,1);
%     scatters( perf([i,4],2:2:end), colors{i} );
%     title('old task'); ylabel('LwF (ours)'); xlabel(mtd_string{imtd});
%     subplot(1,3,2);
%     scatters( perf([i,4],1:2:end), colors{i} );
%     title('new task'); ylabel('LwF (ours)'); xlabel(mtd_string{imtd});
%     subplot(1,3,3);
%     scatterdiffs( perf([i,4],1:2:end), colors{i} );
%     title('new task'); ylabel('difference'); xlabel(mtd_string{imtd}); axis square;

%     print(fullfile(dumpdir, sprintf('scatter_%d_%s.png', imtd, mtd_str{imtd})), '-dpng', '-r240');
% end

% ========== separate bar ========== 
% for imtd=1:3
%     i = mtd_order(imtd);
%     figure(1);
%     clf('reset');
%     set(gcf, 'PaperPosition', [0.25 2.5 8 3]);
%     for p = 1:n_tpairs
%         % suptitle( [titles{p} sprintf('\n')] );
%         tp = tpair_order(p);
%         subplot(2,n_tpairs, p);
%         bars(perf([i,4], tp*2), colors([i,4]));
%         % xlabel('old task');
%         subplot(2,n_tpairs, p+n_tpairs);
%         bars(perf([i,4], tp*2-1), colors([i,4]));
%         % xlabel('new task');
%         % saveas(1,fullfile(dumpdir, [titles{p} '.png']));
%         % print(fullfile(dumpdir, sprintf('%d%d_%s_%s.png', imtd, p, mtd_str{imtd}, titles{p})), '-dpng', '-r300');
%         % waitforbuttonpress;
%     end
%     print(fullfile(dumpdir, sprintf('%d_%s.png', imtd, mtd_str{imtd})), '-dpng', '-r240');
% end

% ========== difference bar ========== 
% for imtd=1:3
%     i = mtd_order(imtd);
%     figure(1);
%     clf('reset');
%     set(gcf, 'PaperPosition', [0.25 2.5 8 2]);
%     subplot(1,2, 1);
%     bardiffs(perf([4,i], 2:2:end), colors{i});
%     title('old task');
%     subplot(1,2, 2);
%     bardiffs(perf([4,i], 1:2:end), colors{i});
%     title('new task');
%     print(fullfile(dumpdir, sprintf('diffbar_%d_%s.png', imtd, mtd_str{imtd})), '-dpng', '-r240');
% end

% ========== difference boxwhiskers ========== 
figure(1);
clf('reset');
set(gcf, 'PaperPosition', [0.25 2.5 8 2]);
for imtd=1:3
    i = mtd_order(imtd);
    subplot(1,3, imtd);
    x = [ perf(4, 2:2:end) - perf(i, 2:2:end); perf(4, 1:2:end) - perf(i, 1:2:end) ]';
    boxwhiskerdiffs(x, colors{i});
    % title('old task');
    % subplot(1,2, 2);
    % boxwhiskerdiffs(perf([4,i], 1:2:end), colors{i});
    % title('new task');
end
print(fullfile(dumpdir, sprintf('diffboxwhiskers.png', imtd, mtd_str{imtd})), '-dpng', '-r240');

% ========== showcase CUB ========== 
% figure(1);
% clf('reset');
% set(gcf, 'PaperPosition', [0.25 2.5 10/3 2]);
% tp = 2;
% subplot(1,2,1);
% bars(perf([mtd_order,4], tp*2), colors([mtd_order,4]));
% % xlabel('old task (ImageNet)');
% ylabel('accuracy');
% subplot(1,2,2);
% bars(perf([mtd_order,4], tp*2-1), colors([mtd_order,4]));
% % xlabel('new task (CUB)');
% print(fullfile(dumpdir, 'summary.png'), '-dpng', '-r360');

function scatters(x,c)
scatter(x(1,:), x(2,:), 'MarkerFaceColor', c,'SizeData',12); hold on;
lh = getlims(x(:), 5);
plot(lh, lh, 'k:');
hold off;
box off;
axis equal;
xlim(lh); ylim(lh);

function scatterdiffs(x,c)
scatter(x(1,:), x(2,:)-x(1,:), 'MarkerFaceColor', c,'SizeData',12); hold on;
lh = getlims(x(:), 5);
plot(lh, [0 0], 'k:');
hold off;
box off;
% axis equal;
xlim(lh); ylim([-5 20]);

function bars(x,c)
assert(numel(x) == numel(c));
for i=1:numel(x)
    bar(i,x(i), c{i}); hold on;
end
ylims = getlims(x,5);
if ylims(2)-ylims(1) > 10
    offset = floor(min(x));
    ylims_ = getlims(x-offset,5);
    ylims_ = ylims_+offset;
    if ylims_(2)-ylims_(1) <= 10
        ylims = ylims_;
    end
end
ylim(ylims);
xlim([0 numel(x)+1.1]);
hold off;

ax = gca;
ax.XTickLabel = {};
ax.YTick = ylims(1):5:ylims(2);
% figureArrow(ax);
box off;

function bardiffs(x,c)
d = x(1,:) - x(2,:);
bar(d, c);
ylim([-5 20]);
xlim([0 numel(d)+1]);
hold off;

ax = gca;
ax.XTickLabel = {};
ax.XTick = [];
ax.XColor = [1 1 1];
% ax.YTick = ylims(1):5:ylims(2);
% figureArrow(ax);
box off;

function boxwhiskerdiffs(x,c)
plot([0.5,2.5], [0,0], 'k'); hold on;
boxplot(x, 'Colors', c, 'Whisker',1.5); hold off;
ylim([-5 20]);
% xlim([0 numel(d)+1]);
% hold off;

ax = gca;
ax.XTickLabel = {'old task', 'new task'};
% ax.XTick = [];
ax.XColor = [1 1 1];
% ax.YTick = ylims(1):5:ylims(2);
% figureArrow(ax);
box off;


function [lim] = getlims(x, grid)
l = min(x); h = max(x);
l = floor(l/grid)*grid;
h = ceil(h/grid)*grid;
if h-l < 10
    l = h-10;
end
lim = [l,h];
