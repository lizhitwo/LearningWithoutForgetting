function [ out ] = chkmkdir( path )
% CHKMKDIR If the PATH directory doesn't exist, make that directory.
% 
% Return:
%   OUT whether the operation is successful (0) or not (otherwise).
% 
% Authors: Zhizhong Li
% 
% See the COPYING file.

out = 0;
if ~exist(path, 'file')
    out = mkdir(path);
end
end

