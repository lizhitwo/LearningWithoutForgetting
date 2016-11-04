function [clswgt] = clsCnt2ClsWeight(clscnt)
% CLSCNT2CLSWEIGHT Given the count for each class, produce the class weights
% so that the weights are proportional to 1/P(w) and the expected weight for
% an example is 1.
% 
% Input:
%   CLSCNT a vector for the sample count for each class
% 
% Return:
%   CLSWGT a vector for the weight for each class
% 
% Authors: Zhizhong Li
% 
% See the COPYING file.

ncls = sum(clscnt~=0);
clswgt = (sum(clscnt)/ncls)./(clscnt + (clscnt==0));