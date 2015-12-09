% --------------------------------------------------------
% CCNN 
% Copyright (c) 2015 [See LICENSE file for details]
% Written by Deepak Pathak, Philipp Krahenbuhl
% --------------------------------------------------------

% Matlab script to generate the labels from classification annotations (same as image-level detection labels) set in VOC
% Run this script and then run the python script to generate hdf5 files : python gen_bb_ind_helper.py

clear all; close all; clc;

out_dir = 'trainList_cl12_seg12';
voc_dir = '/mnt/a/pathak/fcn_mil_cache/VOC2012';
curr_folder = pwd;
cd(voc_dir);

train_seg = textread('./train.txt','%s');
trainval_cl = textread('./ImageSets/Main/trainval.txt','%s');
[train_new, indSeg, indCl] = intersect(train_seg,trainval_cl);

fprintf('Saving output to directory : %s\n',fullfile(voc_dir,out_dir));

fid = fopen(['./' out_dir '/train.txt'],'w');
for i=1:length(train_new)
	fprintf(fid,'%s\n',train_new{i});
end
fclose(fid);

fid = fopen(['./' out_dir '/indicator_train.txt'],'w');
for i=1:length(train_new)
	fprintf(fid,'%s\n',['/mnt/a/pathak/fcn_mil_cache/VOC2012/' out_dir '/ClassIndicator/' train_new{i} '.hf5']);
end
fclose(fid);


classes = { 'background',
    'aeroplane', 
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor' };
tempIndicator = zeros(length(trainval_cl),21);
tempIndicator(:,1) = ones(length(trainval_cl),1);
for i=2:length(classes)
    [~,tempIndicator(:,i)] = textread(['./ImageSets/Main/' classes{i} '_trainval.txt'],'%s %d');
end
tempIndicator(tempIndicator==-1)=0;             % 1: present , -1 : absent , 0 : difficult


indicator = tempIndicator(indCl,:);
dlmwrite(['./' out_dir '/train_labels.txt'], indicator, 'delimiter',' ');

cd(curr_folder);


% ===========================================================================
% Shuffling Code :
% ===========================================================================

clear all; close all; clc;
voc_dir = '/mnt/a/pathak/fcn_mil_cache/VOC2012';
curr_folder = pwd;
cd(voc_dir);

datasetName = 'trainval';
rng(2222);
train_seg = textread(['./' datasetName '_notShuffled.txt'],'%s');
randomSeq = randperm(length(train_seg));

fid = fopen(['./' datasetName '.txt'],'w');
for i=1:length(train_seg)
    fprintf(fid,'%s\n',train_seg{randomSeq(i)});
end
fclose(fid);

fid = fopen(['./indicator_' datasetName '.txt'],'w');
for i=1:length(train_seg)
    fprintf(fid,'%s\n',['/mnt/a/pathak/fcn_mil_cache/VOC2012/SegmentationClassIndicator/' train_seg{randomSeq(i)} '.hf5']);
end
fclose(fid);

% ===========================================================================
% ===========================================================================
