
%% parameters setting  
% clear;
hemi={'lh';'rh'};
hm=2;
surf_tpe='sphere';% or surf_tpe='inflated';
% surf_tpe='inflated';
ssname='litong_xyf';
plotdata = COR_for_plot{1};
load COR_for_plot_new_pearson;
sessionix = 2;
% data_dir = '/home/xianchige/TL/foolDNN_zc';

hemis = {'lh','rh'};
surfsuffix='orig'; %Non-dense surfaces (standard freesurfer outputs); if dense, would be 'DENSETRUNCpt'
[numlh, numrh] = cvnreadsurface(ssname, hemis, 'sphere', surfsuffix, 'justcount',true);


% find the high activated vision voxels in this session
% sessionix = 1;
% beta_threshold = 1;
% betamask = logical(zeros(numlh+numrh,1));
% load(sprintf([data_dir '/session%d/data/GLManalysis_vision/results.mat'],sessionix),'modelmd');
% data = modelmd{2}; %beta_temp = modelmd{2};
% data(find(isnan(data)))  =  -20;
% 
% betamask(find(data>beta_threshold)) = 1;
% betamask_result = zeros(size(betamask));
% 
% %% load the ROI selected vertexes
% labelname={'V1d', 'V2d', 'V3d','V1v','V2v','V3v','V4'};
% for ll = 1:numel(labelname)
%                 temp_label = read_label(ssname,[hemis{1} '.' labelname{ll}]); %  vertices * 5
%                 temp_label_mask1 = zeros(size(betamask));
%                 temp_label_mask1(temp_label(:,1)+1) = 1;
%                 temp_label_mask1 =  betamask & logical(temp_label_mask1); 
%                 temp_label = read_label(ssname,[hemis{2} '.' labelname{ll}]); %  vertices * 5
%                 temp_label_mask2 = zeros(size(betamask));
%                 temp_label_mask2(temp_label(:,1)+1+numlh) = 1;
%                 temp_label_mask2 =  betamask & logical(temp_label_mask2); 
%                 betamask_result = betamask_result | temp_label_mask1 | temp_label_mask2;
%  end

%% show surface figure
[viewpt,~,viewhemis] = cvnlookupviewpoint(ssname,{'lh','rh'},'occip',surf_tpe);
Lookup = [];

label = plotdata(:,31);
noROI = ~(label== 1 | label== 2 | label== 3 | label== 4 | label== 5 | label== 6 | label== 7 | label== 8);
% noROI = label== 0;
data = plotdata(:,5); %edit me %5是RE与AN的相关性searchlight，6是RE与AI的相关性searchlight
% load(sprintf([data_dir '/data/session%d/GLManalysis_vision/results.mat'],sessionix),'modelmd');
% load(sprintf([data_dir '/session%d/data/GLManalysis_vision/results.mat'],sessionix),'modelmd');
% data = modelmd{2};

% data(noROI) = 0; %是否显示非ROI区域

% data(data==1 | data==2 )=1; 
% data(data==3 | data==4 )=2; 
% data(data==5 | data==6 )=3;
% data(data==7 )=4; 
% data(data==8 )=5;

% roi mask
ROI{1} = label==1 | label==2 ;
ROI{2} = label==3 | label==4 ;
ROI{3} = label==5 | label==6 ;
ROI{4} = label==7 ;
ROI{5} = label==8 ;
ROI_name{1}='V1';ROI_name{2}='V2';ROI_name{3}='V3';ROI_name{4}='V4';ROI_name{5}='LO';

tval=valstruct_create(ssname,surfsuffix,data); 
% val = struct('data',<numlh+numrh x 1>,'numlh',numlh,'numrh',numrh);
[rawimg,Lookup,rgbimg] = cvnlookupimages(ssname,tval,viewhemis,viewpt,Lookup,...
    'xyextent',[0.6 0.6],'surftype',surf_tpe,'surfsuffix',surfsuffix,...
    'text',upper(viewhemis),'rgbnan',-20,...
    'clim',[0.05 0.4],'colormap',hot,...
    'roimask',ROI ,'roicolor','w','roiwidth',1,...
    'threshold',0.05); 
figure; 
himg = imshow(rgbimg);
%     'roimask',betamask_result,...

