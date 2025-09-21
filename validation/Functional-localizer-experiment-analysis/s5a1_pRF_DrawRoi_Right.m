
%% parameters setting
clear;
hemi={'lh';'rh'};
%hemi={'rh'};
hm=1;
% surf_tpe='inflated'; %edit me
surf_tpe='sphere';
surfsuffix='orig';

dataDir = '/home/xianchige/YZY/20220523data/sub05/session01/';
%dataDir = '/home/xianchige/YZY/20220322test/data/session1/sub01/';

%% load data
% ssname='litongsubjectfortest';
% ssname='litong_zc1';
 ssname='sub05';
% ssname='litong_dxh';
load('/home/xianchige/YZY/20220523data/sub05/session01/preprocess/result_for_prf_mean/retino_pRF_run_mean.mat');
%load('/home/xianchige/YZY/20220322test/data/session1/preprocess2/result_for_prf/retino_pRF_run06.mat');
% data1=results.ecc;
% data1=results.ang;
data1=results.ang;
data=data1;
% load([dataDir '/surfacedataretino_pRF_run02.mat']);
% data2=results.ang;
% data=data1;%use first prf exp data

%data = (data1+data2)/2;
% data = MRIread([dataDir hemi{hm} '.data.mgz']);
% data = data.vol;

%% load the kastner mask

data_mask = MRIread(['/home/xianchige/YZY/20220523data/sub05/sub05_' hemi{hm} '.Kastner2015Labels'  '.mgz']);
%data_mask = MRIread(['/home/xianchige/YZY/20220322test/data/session1/sub01_' hemi{hm} '.Kastner2015Labels'  '.mgz']);
%data_mask = MRIread(['/home/xianchige/TL/matlabcode/retinotopic_xyf/fsaverageKastner2015Labels-LH.mgz');
data_mask = data_mask.vol;
data_mask_v  =  data_mask';
% data_mask_v(find(data_mask_v~=14 & data_mask_v~=15 & data_mask_v~=1 & data_mask_v~=2 & data_mask_v~=3 & data_mask_v~=4& data_mask_v~=5 & data_mask_v~=6 & data_mask_v~=7)) = 0;
% % data_mask_v(find(data_mask_v~=14 & data_mask_v~=15 )) = 0;
% %data_mask_v(find(data_mask_v~=15)) = 0;


%% 根据之前画好的ROI重新展示，看画得怎么样

% labelname={'V1d', 'V2d', 'V3d','LO1','LO2','V1v','V2v','V3v','V4'};
labelname={'V1d'};
[numlh, numrh] = cvnreadsurface(ssname, hemi, surf_tpe, surfsuffix, 'justcount',true);
betamask_result = logical(zeros(numlh+numrh,1));

        
%% show surface figure
ang_result = valstruct_create(ssname,surfsuffix,data); 
ang_result_c = valstruct_getdata(ang_result,hemi{hm});
%%ecc_result = valstruct_create(ssname,surfsuffix,data1); 
%%ecc_result_c = valstruct_getdata(ecc_result,hemi{hm});

ori=cvnlookupviewpoint(ssname,hemi{hm},'occip',surf_tpe);
clim=[ ];

% %色标卡
% figure; drawcolorbarcircular(cmapang,1);


%%[~,L,rgbimg1]=cvnlookupimages(ssname,ecc_result_c,hemi{hm},ori, ...
    %%'cmap', cmapang(64), ...
   %% 'roimask',data_mask_v ,...
    %%'xyextent',[1 1],'surfsuffix','orig','circulartype',1,...
   %% 'surftype',surf_tpe,'clim',[0 300],'threshold',5);

[~,L,rgbimg2]=cvnlookupimages(ssname,ang_result_c,hemi{hm},ori, ...
    'cmap', cmapang(64), ...
    'roimask',data_mask_v ,...
    'xyextent',[1 1],'surfsuffix','orig','circulartype',1,...
    'surftype',surf_tpe,'clim',[0 330],'threshold',5);
%[~,L,rgbimg1]=cvnlookupimages(ssname,angle_result_c,hemi{hm},ori, ...
   % 'cmap', cmapang(64), ...
    %'roimask',data_mask_v ,...
    %'xyextent',[1 1],'surfsuffix','orig','circulartype',1,...
   % 'surftype',surf_tpe,'clim',[0 300],'threshold',5);


%%figure;
%%h=imshow(rgbimg1);
%%title('ecc')
figure;
h=imshow(rgbimg2);
title('ang')


%% draw lines
% labelname={'V1d', 'V2d', 'V3d','LO1','LO2','V1v','V2v','V3v','V4'};
labelname={ 'V1d'};
R1=zeros(length(ang_result_c),1); %


%press Escape to erase and start again
%double click on final vertex to close polygon
%right click on first vertex, and click "Create mask" to view the result
%Keep going until user closes the window
for i = 1:length(labelname)
    R = [];
    while(ishandle(h))
        [r,rx,ry]=roipoly();
        if(isempty(r))
            continue;
        end
        R=spherelookup_image2vert(r,L)>0;
        
        imgroi=spherelookup_vert2image(R,L,0);
        
        %quick way to merge rgbimg background with roi mask
        tmprgb=bsxfun(@times,rgbimg2,.75*imgroi + .25);
        set(h,'cdata',tmprgb);
        
    end
    % if hm==1
    %     R=R.*(data1(1:nvhm(1))>0);
    % else
    %    R=R.*( data1(1+nvhm(1):end)>0);
    % end
    
    R1=R+R1;
    R1=R1>0;
    
    %% now save the label file
    
    displaysuffix='';
    roiidx=find(R>0);
    
    labelsuffix='';
    
    if(isequal(labelsuffix,'orig'))
        labelsuffix='';
    end
    labelfile=sprintf('%s/%s/label/%s%s.%s.label',cvnpath('freesurfer'),ssname,hemi{hm},displaysuffix,labelname{i});
    
    write_label(roiidx-1,zeros(numel(roiidx),3),ones(numel(roiidx),1),labelfile,ssname,'TkReg');
    
    %% now show the label
    temp_label = read_label(ssname,[hemi{hm} '.' labelname{i}]); %  vertices * 5
    temp_label_mask1 = zeros(size(betamask_result));
    temp_label_mask1(temp_label(:,1)+1+numlh) = 1;
         %temp_label_mask1 =  betamask & logical(temp_label_mask1); 
         %temp_label = read_label(ssname,[hemi{2} '.' labelname{i}]); %  vertices * 5
         %temp_label_mask2 = zeros(size(betamask_result));
         %temp_label_mask2(temp_label(:,1)+1+numlh) = 1;
         %temp_label_mask2 =  betamask & logical(temp_label_mask2); 
         %betamask_result = betamask_result |temp_label_mask1 | temp_label_mask2;
    betamask_result = betamask_result |temp_label_mask1 ;
         
   
    [viewpt,~,viewhemis] = cvnlookupviewpoint(ssname,hemi{hm},'occip',surf_tpe);
    Lookup = [];

  % val = struct('data',<numlh+numrh x 1>,'numlh',numlh,'numrh',numrh);
    [rawimg,Lookup,rgbimg] = cvnlookupimages(ssname,ang_result,viewhemis,viewpt,Lookup,...
            'cmap', cmapang(64), ...
            'roimask',betamask_result ,...
            'xyextent',[1 1],'surfsuffix','orig','circulartype',1,...
            'surftype',surf_tpe,'clim',[0 330],'threshold',5);
    
    figure; 
    h = imshow(rgbimg);
    %% draw another roi
end
