function [resultfile] = visualizeTest(lastfc_out, varargin)
% VISUALIZETEST   Compile test set online evaluation submissions, and visualize them
% 
% Authors: Zhizhong Li
% 
% See the COPYING file.

% usage examples:

% VOCnokeep = load('/home/zhizhong/Dataset/dump/Paper_all_grid_search/Alex_Places2_VOCtrval_fork_x_redo_x_partial_x_nokeep/fork_1_redo_1_partial_tr_1.00/info.mat');
% lastfc_out = VOCnokeep.info.valall_NewTask.val.lastfc_out{1};
% visualizeTest( lastfc_out, 'set', 'VOC', 'path_dump', '/home/zhizhong/SSD_Dataset/VOC2012/VOCdevkit/results/VOC2012/', 'write', true);

% VOClwf = load('/home/zhizhong/Dataset/dump/Paper_all_grid_search/Alex_Places2_VOCtrval_fork_x_redo_x_partial_x_lockkeep_TEMP2/fork_1_redo_1_partial_tr_1.00/info.mat');
% lastfc_out = VOClwf.info.valall_NewTask.val.lastfc_out{1};
% visualizeTest( lastfc_out, 'set', 'VOC', 'path_dump', '/home/zhizhong/SSD_Dataset/VOC2012/VOCdevkit/results/VOC2012/', 'write', true);

% VOCasfeat = load('/home/zhizhong/Dataset/dump/Paper_all_grid_search/Alex_Places2_VOCtrval_fork_x_redo_x_partial_x_asfeat_5lr/forkredo_1_new_1_partial_tr_1.00/info.mat');
% lastfc_out = VOCasfeat.info.valall_NewTask.val.lastfc_out{1};
% visualizeTest( lastfc_out, 'set', 'VOC', 'path_dump', '/home/zhizhong/SSD_Dataset/VOC2012/VOCdevkit/results/VOC2012/', 'write', true);

% VOCjoint = load('/home/zhizhong/Dataset/dump/Paper_all_grid_search/Alex_Places2_VOCtrval_fork_x_redo_x_partial_x_joint/fork_1_redo_1_partial_tr_1.00/info.mat');
% lastfc_out = VOCjoint.info.valall_NewTask.val.lastfc_out{1};
% visualizeTest( lastfc_out, 'set', 'VOC', 'path_dump', '/home/zhizhong/SSD_Dataset/VOC2012/VOCdevkit/results/VOC2012/', 'write', true);

% Places2nokeep = load('/home/zhizhong/Dataset/dump/Paper_all_grid_search/Alex_Places2_VOCtrval_fork_x_redo_x_partial_x_nokeep/fork_1_redo_1_partial_tr_1.00/Places2_test_info.mat');
% lastfc_out = Places2nokeep.info.val.lastfc_out{1};
% visualizeTest( lastfc_out, 'set', 'Places2', 'path_dump', '/home/zhizhong/SSD_Dataset/Places2/Places2_devkit/results/', 'write', true);

% Places2lwf = load('/home/zhizhong/Dataset/dump/Paper_all_grid_search/Alex_Places2_VOCtrval_fork_x_redo_x_partial_x_lockkeep_TEMP2/fork_1_redo_1_partial_tr_1.00/Places2_test_info.mat');
% lastfc_out = Places2lwf.info.val.lastfc_out{1};
% visualizeTest( lastfc_out, 'set', 'Places2', 'path_dump', '/home/zhizhong/SSD_Dataset/Places2/Places2_devkit/results/', 'write', true);

% Places2asfeat = load('/home/zhizhong/Dataset/dump/Paper_all_grid_search/Alex_Places2_VOCtrval_fork_x_redo_x_partial_x_asfeat_5lr/forkredo_1_new_1_partial_tr_1.00/Places2_test_info.mat');
% lastfc_out = Places2asfeat.info.val.lastfc_out{1};
% visualizeTest( lastfc_out, 'set', 'Places2', 'path_dump', '/home/zhizhong/SSD_Dataset/Places2/Places2_devkit/results/', 'write', true);

% Places2joint = load('/home/zhizhong/Dataset/dump/Paper_all_grid_search/Alex_Places2_VOCtrval_fork_x_redo_x_partial_x_joint/fork_1_redo_1_partial_tr_1.00/Places2_test_info.mat');
% lastfc_out = Places2joint.info.val.lastfc_out{1};
% visualizeTest( lastfc_out, 'set', 'Places2', 'path_dump', '/home/zhizhong/SSD_Dataset/Places2/Places2_devkit/results/', 'write', true);


opts.set = 'Places2'; % which dataset's test set?
opts.path_dump = '../dump/testresults';
opts.write = false; % overwrite existing reports?
opts = vl_argparse(opts, varargin);

p = getPath('path_dump', opts.path_dump);


assert(ndims(lastfc_out)==2);
% keyboard;

switch opts.set
    case 'Places2'
        assert(size(lastfc_out, 1) == 401);
        % name of classes
        classfile = fullfile(p.path_Placesmeta, 'categories.txt');
        fid = fopen(classfile, 'r');
        classRead = textscan(fid, '%s %d');
        fclose(fid);
        classNames = classRead{1};
        classID = classRead{2};
        assert(all(classID == (0:numel(classID)-1)'));

        % name of images
        imgfilefile = fullfile(p.path_Placesmeta, 'test.txt');
        fid = fopen(imgfilefile, 'r');
        fileRead = textscan(fid, '%s');
        fclose(fid);

        % classification results
        [confidences,predictions] = sort(lastfc_out, 1, 'descend') ;

        % these variables are used in visualization
        top5 = predictions(1:5,:)-1;
        top5scores = confidences(1:5,:);
        ID2names = containers.Map(num2cell(classID), classNames);
        imdir = fullfile(p.path_Placesroot, 'test/');
        fileNames = fileRead{1};

        % write to results
        assert(size(lastfc_out, 2) == numel(fileNames));
        if opts.write
            table = cell2table( [ fileNames, num2cell(top5') ] );
            resultfile = fullfile(opts.path_dump, 'Places2.test.pred.txt');
            writetable(table, resultfile, 'Delimiter', ' ', 'WriteVariableNames', false);
        end

    case 'VOC'
        assert(size(lastfc_out, 1) == 20);
        sz = size(lastfc_out); lastfc_out = reshape(vl_nnsigmoid(reshape(lastfc_out, [1 1 sz])), sz);

        devkitpath = fullfile(p.path_VOCroot, 'VOCdevkit/');
        VOCopts = load(fullfile(devkitpath, 'VOCopts.mat'));
        classNames = VOCopts.classes;

        % name of images
        imgfilefile = fullfile(p.path_VOCmeta, 'test.txt');
        fid = fopen(imgfilefile, 'r');
        fileRead = textscan(fid, '%s');
        fileIDs = fileRead{1};
        fclose(fid);

        % classification results
        [confidences,predictions] = sort(lastfc_out, 1, 'descend') ;

        % these variables are used in visualization
        top5 = predictions(1:5,:);
        top5scores = confidences(1:5,:);
        ID2names = containers.Map(num2cell(1:20), classNames);
        imdir = p.path_VOCimdir;
        fileNames = strcat(fileIDs, '.jpg');

        % write to results
        assert(size(lastfc_out, 2) == numel(fileNames));
        if opts.write
            resultfile_pattern = fullfile(p.path_dump, 'Main/%s_cls_test_%s.txt');
            for i=1:VOCopts.nclasses
                cls = VOCopts.classes{i};
                resultfile = sprintf(resultfile_pattern,'comp2',cls);
                % table = cell2table( [ fileNames, num2cell(lastfc_out(i,:)') ] );
                fid = fopen(resultfile, 'wt');
                for l=1:numel(fileIDs)
                    fprintf(fid, '%s %f\n', fileIDs{l}, lastfc_out(i,l));
                end
                fclose(fid);
                % writetable(table, resultfile, 'Delimiter', ' ', 'WriteVariableNames', false);
                % fprintf(fid,'%s %f\n',ids{i},c);
            end
        end
    otherwise
        assert(false, 'opts.set unrecognized');
end



% Visualize images
n_img = numel(fileNames);
iimg = 1;
while iimg <= n_img
    figure(1);
    imgfile = [ imdir fileNames{iimg} ];
    im = imread(imgfile);
    subplot(1,2,1); imshow(im); title(fileNames{iimg});
    subplot(1,2,2); fileNames{iimg}
    top5clsnames = values(ID2names, num2cell(top5(:,iimg)));
    for i=1:5
        txt{i} = sprintf('%s: %.2f', top5clsnames{i}, top5scores(i,iimg));
    end
    plot(ones(5), 'x');
    legend(txt);


    k = 0;
    while k==0
        try
            k = waitforbuttonpress;
        catch ME
            k = -1;
        end
        if k==1
            f = gcf;
            keypress = f.CurrentCharacter;
            % ascii's 28:31 are for <- -> ^ v
            switch double(keypress)
                case {28,30}
                    iimg = iimg-1;
                    if iimg<=0, iimg = 1; end
                case {29,31}
                    iimg = iimg+1;
                otherwise
                    % keys{iimg} = keypress;
            end
        end
    end
    if k==-1, return; end
end