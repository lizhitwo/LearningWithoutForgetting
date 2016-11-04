function [ret] = Pushme(msgtxt, details)
% PUSHME   Ring notification using the BoxCar app.
%   Input:
%     MSGTXT the headline text
%     DETAILS the content text
% 
% Authors: Zhizhong Li
% 
% See the COPYING file.

if nargin < 2
    details = 'N/A';
end
assert(isstr(msgtxt) && isstr(details));

if strcmp(getenv('USER'), 'zhizhong')
    % IMPORTANT: Replace the token with yours or *I* get your message which can be a bad thing for both of us
    token = '[bleep]'; % this is the token that ***you*** get from installing Boxcar 2 app. Should be something like a 20-char long alphanumeric code
else
    warning('Push notification: please use your own token for push notification. Skipping for now...');
    return;
end

details_markdown = details;
details_markdown=strrep(details_markdown,'>','&gt;');
details_markdown=strrep(details_markdown,'/','&#47;');
details_markdown=strrep(details_markdown,'<','&lt;');
details_markdown=strrep(details_markdown,sprintf('\n'),'<br/>');

% IMPORTANT: Replace the token with yours or *I* get your message which can be a bad thing for both of us
postcontent = {'user_credentials'; token;
    'notification[title]'; ['cashew: ' msgtxt];
    'notification[long_message]'; ['<b>' details_markdown '</b>'];
    };

urlread('https://new.boxcar.io/api/notifications', 'Post', postcontent);

fprintf('Message pushed: %s\n%s\n', msgtxt, details);
