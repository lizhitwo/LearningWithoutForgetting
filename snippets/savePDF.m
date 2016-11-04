function savePDF(hfigure, filename)
% SAVEPDF   Save figure to pdf.
% 
% Authors: Zhizhong Li
% 
% See the COPYING file.

set(hfigure,'Units','Inches');

pos = get(hfigure,'Position');

set(hfigure,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);

print(hfigure,filename,'-dpdf','-r0');
