function figureArrow(ax)
% FIGUREARROW  add arrows to a plot.
%   Input:
%     AX the axes for adding the arrow.

% Authors: Zhizhong Li
% 
% See the COPYING file.

% Adjusted from http://stackoverflow.com/questions/17047371/how-to-get-arrows-on-axes-in-matlab-plot

ca = gca;
axes(ax);
axp = ax.Position;

% xs=axp(1);
% xe=axp(1)+axp(3)+0.02;
% ys=axp(2);
% ye=axp(2)+axp(4)+0.02;

% annotation('arrow', [xs xe],[ys ys]);
% annotation('arrow', [xs xs],[ys ye]);

x0=axp(1);
xs=axp(1)+axp(3);
xe=axp(1)+axp(3)+0.01;
y0=axp(2);
ys=axp(2)+axp(4);
ye=axp(2)+axp(4)+0.01;

annotation('arrow', [xs xe],[y0 y0]);
annotation('arrow', [x0 x0],[ys ye]);

box off;
% set(ax,'YColor',get(ax,'Color'));
% set(ax,'XColor',get(ax,'Color'));

axes(ca);