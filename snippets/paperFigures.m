flag_TRDATA = 1;
flag_ACCUMU = 1;


fs = 14;
fn = 'Times-Roman';
    colors = {'r','g','c','b'};
    markers = {'s','o','^','x'};
    % xoffset = [-0.21 -0.07 0.07 0.21];
    xoffset = [-0.15 -0.05 0.05 0.15];

%% TRDATA!
if flag_TRDATA
    TRDATA = cat(3, [71.9044   41.5212 67.6619 39.207  61.8005 39.2668 53.9507 37.6858
    72.2998 43.2469 68.369  43.1521 62.5978 42.9377 53.8645 42.9077
    68.1923 43.1372 65.0865 43.1372 55.9209 43.1372 37.6831 43.1372
    72.0956 43.2219 68.6173 42.5287 62.8662 42.5387 54.0559 41.8055], ...
    [71.9173    41.5411 66.3776 39.9601 60.3299 39.4713 49.7129 38.0598
    72.1583 43.0623 67.3872 43.2469 60.9684 43.3367 49.894  42.9526
    68.1689 43.1372 64.2383 43.1372 54.1941 43.1372 38.0699 43.1372
    72.0825 43.3117 67.6974 43.0274 61.9097 42.793  50.0883 42.2693], ...
    [72.0297    41.4663 67.3934 40.4289 61.7488 38.3292 49.5901 38.3541
    72.1897 43.3017 67.9935 42.9825 62.7962 42.6883 49.346  43.0424
    68.1676 43.1372 65.2561 43.1372 55.1604 43.1372 36.2425 43.1372
    72.1202 43.3416 68.4501 43.4065 63.0153 42.793  49.3868 42.6883]);


    if ishandle(101), close(101); end
    hfigure1 = figure(101); 
    pos = hfigure1.Position; hfigure1.Position = [ pos(1) pos(2) 400 370 ];

    TRDATA_new = TRDATA(:, 1:2:end, :);
    sz = size(TRDATA_new);
    for imethod = 1:sz(1)
        Y = reshape(TRDATA_new(imethod,:,:), sz(2), sz(3));
        X = repmat((1:sz(2))', 1, sz(3));
        plot(mean(X,2),mean(Y,2), ['-' colors{imethod}]); hold on;
        X = X + xoffset(imethod);
        hline(imethod) = plot(X(:),Y(:), [markers{imethod} colors{imethod}], 'linewidth', 1, 'markers', 6, 'MarkerFaceColor', [colors{imethod}]);
    end
    hold off; xlim([0.5 4.5]);
    ax1 = gca; ax1.XTick = 1:4; ax1.XTickLabel = {'100%', '30%', '10%', '3%'};  ax1.FontSize = fs; ax1.FontName = fn;
    % title('(a) New task (VOC)', 'FontSize', fs, 'FontName', fn);
    grid on;

    if ishandle(102), close(102); end
    hfigure2 = figure(102); 
    pos = hfigure2.Position; hfigure2.Position = [ pos(1) pos(2) 400 370 ];

    TRDATA_old = TRDATA(:, 2:2:end, :);
    sz = size(TRDATA_old);
    for imethod = 1:sz(1)
        Y = reshape(TRDATA_old(imethod,:,:), sz(2), sz(3));
        X = repmat((1:sz(2))', 1, sz(3));
        plot(mean(X,2),mean(Y,2), ['-' colors{imethod}]); hold on;
        X = X + xoffset(imethod);
        hline(imethod) = plot(X(:),Y(:), [markers{imethod} colors{imethod}], 'linewidth', 1, 'markers', 6, 'MarkerFaceColor', [colors{imethod}]);
    end
    hold off; xlim([0.5 4.5]);
    ax2 = gca; ax2.XTick = 1:4; ax2.XTickLabel = {'100%', '30%', '10%', '3%'};  ax2.FontSize = fs; ax2.FontName = fn;
    % title('(b) Old task (Places2)', 'FontSize', fs, 'FontName', fn);
    grid on;

    if ishandle(103), close(103); end
    hfigure3 = figure(103); 
    pos = hfigure3.Position; hfigure3.Position = [ pos(1) pos(2) 220 370 ];

    % for imethod = 1:sz(1), plot(0,0,['^' colors{imethod}]); hold on; end
    for imethod = 1:sz(1)
        hline(imethod) = plot(X(:),Y(:), [markers{imethod} colors{imethod}], 'linewidth', 1, 'markers', 6, 'MarkerFaceColor', [colors{imethod}]); hold on;
        hline(imethod).Visible = 'off';
    end
    hlegend = legend(hline, {'fine-tuning', 'joint training', 'feat. extraction', 'LwF (ours)'}, 'FontSize', fs, 'FontName', fn);
    set(hlegend, 'Position', [0.15 0.7 0.7 0.16], 'Units', 'normalized');
    ax3 = gca; 
    axis off;

    % pos = ax3.Position; ax3.Position = [0.85 pos(2) 0.1 pos(4)];
    % pos = ax2.Position; ax2.Position = [0.40 pos(2) 0.27 pos(4)];
    % pos = ax1.Position; ax1.Position = [0.07 pos(2) 0.27 pos(4)];

    figureArrow(ax1);
    figureArrow(ax2);

    savePDF(hfigure1, '/home/zhizhong/Dataset/dump/TRDATA1.pdf');
    savePDF(hfigure2, '/home/zhizhong/Dataset/dump/TRDATA2.pdf');
    savePDF(hfigure3, '/home/zhizhong/Dataset/dump/TRDATA3.pdf');
end


%% ACCUMU!
if (flag_ACCUMU)
  for figname = {'VOC', 'MIT', 'legend'}
    figname = figname{1};
    switch figname
    case {'VOC','legend'}
        ACCUMU = cat(3, [43.1372 41.4214 82.502  39.611  82.0256 71.4116 38.1446 81.3782 69.9911 62.1545
        nan nan nan nan nan nan 43.2469 82.9092 71.5658 62.6691
        nan nan nan nan nan nan 43.1372 79.6786 65.7831 59.9183
        43.1372 43.4264 82.4787 42.8928 82.569  71.3537 42.4439 82.3906 71.0518 62.6586], ...
        [43.1372    41.2269 82.5046 39.7756 82.0246 71.5347 38.2743 81.4148 70.0692 61.8
        nan nan nan nan nan nan 43.0623 83.0404 71.3047 62.4142
        nan nan nan nan nan nan 43.1372 79.6774 65.7124 59.9357
        43.1372 43.3815 82.4756 42.9626 82.5504 71.3542 42.3392 82.3409 71.0193 62.62], ...
        [43.1372    41.3815 82.4492 39.7806 81.9457 71.3519 38.788  81.3918 69.9521 62.213
        nan nan nan nan nan nan 43.3017 82.9902 71.414  62.4235
        nan nan nan nan nan nan 43.1372 79.6422 65.7182 59.9589
        43.1372 43.3716 82.534  42.9027 82.6142 71.4474 42.6933 82.368  71.1332 62.7013]);
    case 'MIT'
        ACCUMU = cat(3, [57.032 53.74   74.4472 50.51   72.973  68.4105 47.452  71.9902 65.5936 69.9541
        nan nan nan nan nan nan 55.44   74.2015 71.831  70.4128
        nan nan nan nan nan nan 57.032  73.2187 68.4105 67.8899
        57.032  56.138  74.4472 55.202  75.43   71.0262 54.094  73.7101 69.0141 70.1835], ...
        [57.032 53.992  74.6929 50.47   72.973  68.4105 47.69   72.973  67.4044 69.2661
        nan nan nan nan nan nan 55.374  75.9214 70.6237 72.0183
        nan nan nan nan nan nan 57.032  73.2187 67.8068 68.578
        57.032  56.136  76.4128 55.156  74.9386 70.2213 53.942  73.7101 67.8068 70.8716], ...
        [57.032 53.794  74.2015 50.184  72.2359 68.4105 47.536  72.973  65.3924 68.8073
        nan nan nan nan nan nan 55.49   75.1843 69.8189 70.1835
        nan nan nan nan nan nan 57.032  72.4816 67.6056 68.8073
        57.032  56.12   75.43   55.084  74.2015 71.6298 53.8    72.973  69.4165 72.0183]);
    end
    assert(size(ACCUMU, 2) == 10);
    ACCUMU(:,11,:) = nan;

    ACCUMU_avg = mean(ACCUMU, 3);
    ACCUMU_std = std(ACCUMU, 0, 3);
    taskcol = [ 1 2 4 7; 11 3 5 8; 11 11 6 9; 11 11 11 10];

    if ishandle(111), close(111); end
    hfigure = figure(111); 
    pos = hfigure.Position; hfigure.Position = [ pos(1) pos(2)-200 470 550 ];

    ax = {};
    switch figname
        case 'VOC'
            ylbl = {'Places2', 'VOC (part 1)', 'VOC (part 2)', 'VOC (part 3)'};
            xlbl = {'Places2', '  VOC\newline(part 1)', '  VOC\newline(part 2)', '  VOC\newline(part 3)'};
        case 'MIT'
            ylbl = {'ImageNet', 'Scenes (part 1)', 'Scenes (part 2)', 'Scenes (part 3)'};
            xlbl = {'Image-\newline  Net', 'Scenes\newline(part 1)', 'Scenes\newline(part 2)', 'Scenes\newline(part 3)'};
        otherwise
            ylbl = {'', '', '', ''};
            xlbl = {'', '', '', ''};
    end
    for itask = 1:size(taskcol,1)
        ax{itask} = subplot(4,1,itask);
        for imethod = 1:size(ACCUMU,1)
            avg = ACCUMU_avg( imethod, taskcol(itask,:) );
            err = ACCUMU_std( imethod, taskcol(itask,:) );
            X = (1:numel(avg));
            hline_(imethod) = plot(X,avg, ['-' colors{imethod}], 'linewidth', 1, 'markers', 6, 'MarkerFaceColor', [colors{imethod}]); hold on;
            X = X + xoffset(imethod);
            hline(imethod) = plot(X,avg, [markers{imethod} colors{imethod}], 'linewidth', 1, 'markers', 6, 'MarkerFaceColor', [colors{imethod}]);
            herr(imethod) = errorbar(X,avg, err*2, 'LineStyle', 'none', 'Color', [0 0 0]);
            % hline(imethod) = plot(X,avg, ['-' colors{imethod}], 'linewidth', 1);
            if strcmp(figname, 'legend'), hline_(imethod).Visible = 'off'; hline(imethod).Visible = 'off'; herr(imethod).Visible = 'off'; end
        end
        ymax = ACCUMU_avg(:, taskcol(itask,:)); ymin = floor(min(ymax(:))/5)*5; ymax = ceil(max(ymax(:))/5)*5;
        hold off; xlim([0.5 4.5]); ylim([ymin ymax]);
        ylabel(xlbl{itask}, 'FontSize', fs, 'FontName', fn);
        ax1 = gca; ax1.XTick = 1:4; % cellfun(@sprintf, {'Task #1', '+ #2', '+ #3', '+ #4'}, 'UniformOutput', false); 
        if itask==size(taskcol,1), ax1.XTickLabel = xlbl; ax1.FontSize = fs; ax1.FontName = fn;
        else ax1.XTickLabel = {}; ax1.FontSize = fs; ax1.FontName = fn; end

        % if itask==1, title('Task performance'); end
        grid on;
        box off;
        % ax = gca; pos = ax.Position; ax.Position = [0.1 pos(2) 0.6 pos(4)];
    end

    if strcmp(figname, 'legend')
        for itask=1:size(taskcol,1)
            subplot(4,1,itask);
            ax = gca;
            ax.Visible = 'off';
        end
        % for imethod = 1:sz(1), plot(0,0,['^' colors{imethod}]); hold on; end
        pos = hfigure.Position; hfigure.Position = [ pos(1) pos(2)-200 220 550 ];
        hlegend = legend(hline, {'fine-tuning', 'joint training', 'feat. extraction', 'LwF (ours)'}, 'FontSize', fs, 'FontName', fn);
        set(hlegend, 'Position', [0.15 0.7 0.7 0.16], 'Units', 'normalized');
    end
    savePDF(hfigure, sprintf('/home/zhizhong/Dataset/dump/ACCUMU_%s.pdf', figname));
  end
end