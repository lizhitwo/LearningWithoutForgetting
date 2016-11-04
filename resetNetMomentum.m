function [ret] = resetNetMomentum(net, varargin)
% RESETNETMOMENTUM   from a (loaded, etc.) network, delete/reset optimizer state.
%   Input:
%     NET a struct or a saved .mat file name with a NET variable in it
%       If it's a string, then overwrite the file after resetting.
%   Output:
%     RET reset net
%   Options:
%     See code comments
% 
% Authors: Zhizhong Li
% 
% See the COPYING file.
opts.fields = {'momentum', 'momentum2', 'updateCount'}; % which fields of the net variable is of concern.
opts.mode = 'reset';    % 'reset' for making all zeros; 'delete' for removing fields
opts = vl_argparse(opts, varargin);

flag_isfile = isstr(net);
if flag_isfile
    net_load = load(net); ret = net_load.net;
else
    ret = net;
end

for i=1:numel(ret.layers)
    for f = opts.fields
        f = f{1};
        if isfield(ret.layers{i}, f)
          switch opts.mode
          case 'reset'
            if iscell(ret.layers{i}.(f))
                for j=1:numel(ret.layers{i}.(f))
                    ret.layers{i}.(f){j} = ret.layers{i}.(f){j}*0;
                end
            else
                ret.layers{i}.(f) = ret.layers{i}.(f)*0;
            end
          case 'remove'
            ret.layers{i} = rmfield(ret.layers{i}, f);
          end
        end
    end
end

if flag_isfile
    net_load.net = ret;
    save(net, '-struct', 'net_load', '-v7.3');
end
