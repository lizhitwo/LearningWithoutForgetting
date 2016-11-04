function [p] = getPath( varargin )
% GETPATH   get the path / file name for all related models / dataset /
% annotations / dump files / etc
%   Options:
%     See code comments
% 
% Authors: Zhizhong Li
% 
% See the COPYING file.

opts.path_dump = '../dump/'; % the results dumping directory
opts = vl_argparse(opts, varargin);

% path to this code
p.path_exeroot = fileparts(mfilename('fullpath'));
% path to datasets' root
p.path_dsetroot = '/home/zhizhong/SSD_Dataset/';
% the results dumping directory
p.path_dump = opts.path_dump;
chkmkdir( p.path_dump );
% cache directory
p.path_cacheroot = '../cache/';
chkmkdir( p.path_cacheroot );

% -------------------------------------------------------------------------
% ImageNet ILSVRC stuff
% -------------------------------------------------------------------------
p.path_imgroot = fullfile(p.path_dsetroot, 'ILSVRC2012/');
p.path_imgtrain = fullfile(p.path_imgroot, 'ILSVRC2012_img_train/');
p.files_imgtrain_synsets = dir(fullfile(p.path_imgtrain, 'n*')); p.files_imgtrain_synsets = {p.files_imgtrain_synsets.name};
p.path_imgval_relative_to_imgtrain = '../ILSVRC2012_img_val/';
p.path_imgval = fullfile(p.path_imgtrain, p.path_imgval_relative_to_imgtrain);
p.file_imgvalgt = fullfile(p.path_imgroot, 'ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt');
p.file_imgmeta = fullfile(p.path_imgroot, 'ILSVRC2012_devkit_t12/data/meta.mat');
% cache for ImageNet imdb (because this one takes time to compute)
p.path_imgIMDB.train = fullfile(p.path_cacheroot, 'IMDB_train.mat');
p.path_imgIMDB.val = fullfile(p.path_cacheroot, 'IMDB_val.mat');

% -------------------------------------------------------------------------
% Pretrained network stuff
% -------------------------------------------------------------------------
p.file_caffe_alex = fullfile(p.path_exeroot, 'imagenet-caffe-alex.mat') ;
p.file_caffe_alex_places2 = fullfile(p.path_exeroot, 'convertcaffe/alexnet_places2_iter_158165_convert.mat') ; % alexnet_places2_convert
p.file_caffe_vgg = fullfile(p.path_exeroot, 'imagenet-vgg-verydeep-16.mat');

% and the responses for different new task images for the pretrained network.
nets = {'Alex_ImNet', 'Alex_Places2', 'VGG_ImNet'};
datasets = {'VOC', 'CUB', 'MIT67', 'VOCtrval', 'MNIST'};
for inet = 1:numel(nets)
    for idtset = 1:numel(datasets)
        % e.g. p.path_response.Alex_ImNet.VOC = fullfile(p.path_cacheroot, 'Alex_ImNetVOCresponse_train.mat');
        p.path_response.(nets{inet}).(datasets{idtset}).train = fullfile(p.path_cacheroot, [ nets{inet} datasets{idtset} 'response_train.mat']);
    end
    p.path_response.(nets{inet}).VOC.trainval = p.path_response.(nets{inet}).VOCtrval.train;
end

% [ HACK ] NOTE: this path_OrigNetresponse field should be given the value of p.path_response.(network).(dataset) according to the situation!
p.path_OrigNetresponse = p.path_response.Alex_ImNet; 

% -------------------------------------------------------------------------
% VOC stuff
% -------------------------------------------------------------------------
p.path_VOCroot = fullfile(p.path_dsetroot, 'VOC2012/');
p.path_VOCmeta = fullfile(p.path_VOCroot, 'ImageSets/Main/');
p.path_VOCimdir = fullfile(p.path_VOCroot, 'JPEGImages/');

% -------------------------------------------------------------------------
% CUB stuff
% -------------------------------------------------------------------------
p.path_CUBroot = fullfile(p.path_dsetroot, 'CUB_200_2011/');
p.path_CUBmeta = fullfile(p.path_CUBroot, 'lists/');
p.path_CUBimdir = fullfile(p.path_CUBroot, 'images/');

% -------------------------------------------------------------------------
% MIT indoor scenes stuff
% -------------------------------------------------------------------------
p.path_MIT67root = fullfile(p.path_dsetroot, 'MIT_67_indoor/');
p.path_MIT67meta = p.path_MIT67root;
p.path_MIT67imdir = fullfile(p.path_MIT67root, 'Images/');

% -------------------------------------------------------------------------
% MNIST stuff
% -------------------------------------------------------------------------
p.path_MNISTroot = fullfile(p.path_dsetroot, 'MNIST/');
p.path_MNISTimdir = 'imdb.img_MNIST';

% -------------------------------------------------------------------------
% Places2 stuff
% -------------------------------------------------------------------------
p.path_Placesroot = fullfile(p.path_dsetroot, 'Places2/');
p.path_Placesmeta = fullfile(p.path_Placesroot, 'Places2_devkit/');

end
