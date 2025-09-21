
%% parameters setting
clear;
hemi={'lh';'rh'};
hm=1;%editme 1-L 2-R
surf_tpe='inflated'; %edit me
% surf_tpe='sphere';
dataDir = '/home/xianchige/YZY/20220523data/sub01/session01/GLManalysis';

%% load data
ssname='sub01';
interptype = 'linear';  % what type of interpolation?
surfsuffix='orig';
[numlh, numrh] = cvnreadsurface(ssname, hemi, surf_tpe, surfsuffix, 'justcount',true);

load([dataDir '/1back/results.mat']);
%data1=results.ang;
% load([dataDir '/surfacedataretino_pRF_run02.mat']);
% data2=results.ang;
%data=data1;
% beta_diff1 = modelmd{2}(:,1);
%beta_diff2 = modelmd{2}(:,2);
%beta_diff3 = modelmd{2}(:,3);
beta_diff1 = modelmd{2}(:,1)-modelmd{2}(:,2);
%  beta_diff1 = modelmd{2}(:,1)-modelmd{2}(:,3);


%%beta_diff1(find(beta_diff1<beta_threshold2)) = -100;
% beta_diff2(find(beta_diff2<beta_threshold1)) = -100;
% beta_diff3(find(beta_diff1<beta_threshold2)) = -100;
% beta_diff4(find(beta_diff2<beta_threshold2)) = -100;
% beta_diff2 = modelmd{2}(:,3)-modelmd{2}(:,2);
% beta_diff3 = modelmd{2}(:,3);
% beta_diff4 = modelmd{2}(:,3)-modelmd{2}(:,4);
% beta_diff5 = modelmd{2}(:,3)-modelmd{2}(:,5);

beta_threshold=0;%L-0.7 R-2
betamask = logical(zeros(numlh+numrh,1));
betamask(find(beta_diff1>=beta_threshold))=2;%逻辑函数，虽然设的是2,其实是令对应体素的�?�?
%betamask_result = betamask;
betamask_result=zeros(size(betamask));

%beta_diff1(find(beta_diff1<beta_threshold_LO)) = -100;

%% load the kastner mask
data_mask = MRIread(['/home/xianchige/YZY/20220523data/sub01/sub01_' hemi{hm} '.Kastner2015Labels'  '.mgz']);
data_mask = data_mask.vol;
data_mask_v  =  data_mask';
%%ata_mask_v(find(data_mask_v~=14 & data_mask_v~=15 & data_mask_v~=1 & data_mask_v~=2 & data_mask_v~=3 & data_mask_v~=4& data_mask_v~=5 & data_mask_v~=6 & data_mask_v~=7)) = 0;

%% load the ROI selected vertexes
labelname={'V1d','V2d','V3d','V1v','V2v','V3v','V4','IOG','pFus','mFus'};
%%labelname={'V1d','V2d','V3d','V1v','V2v','V3v','V4','IOG','mFus','pFus'};
% % % % % % for ll = 1:numel(labelname)
% % % % % %                 temp_label = read_label(ssname,[hemi{1} '.' labelname{ll}]); %  vertices * 5
% % % % % %                 temp_label_mask1 = zeros(size(betamask));
% % % % % %                 temp_label_mask1(temp_label(:,1)+1) = 1;
% % % % % % %                 temp_label_mask1 =  betamask & logical(temp_label_mask1); 
% % % % % %                 temp_label = read_label(ssname,[hemi{2} '.' labelname{ll}]); %  vertices * 5
% % % % % %                 temp_label_mask2 = zeros(size(betamask));
% % % % % %                 temp_label_mask2(temp_label(:,1)+1+numlh) = 1;
% % % % % % %                 temp_label_mask2 =  betamask & logical(temp_label_mask2); 
% % % % % %                 betamask_result = betamask_result |temp_label_mask1 | temp_label_mask2;
% % % % % % end
betamask_result = valstruct_create(ssname,surfsuffix,betamask_result); 
betamask_result_a = valstruct_getdata(betamask_result,hemi{hm});
%% show surface figure
%%[viewpt,~,viewhemis] = cvnlookupviewpoint(ssname,{'lh','rh'},'occip',surf_tpe);
[viewpt,~,viewhemis] =cvnlookupviewpoint(ssname,hemi{hm},'occip',surf_tpe); %editme
%%[viewpt,~,viewhemis] = cvnlookupviewpoint(ssname,hemi{hm},'occip',surf_tpe);
Lookup = [];

%diff4
tval4=valstruct_create(ssname,surfsuffix,beta_diff1);
%tval4=valstruct_create(ssname,surfsuffix,beta_diff1); 
val4 = valstruct_getdata(tval4,hemi{hm});

% [rawimg,Lookup,rgbimg] = cvnlookupimages(ssname,val4,viewhemis,viewpt,Lookup,...
%     'xyextent',[1 1],'surftype',surf_tpe,'surfsuffix',surfsuffix,...
%     'text',upper(viewhemis),'rgbnan',-10,...
%     'roimask',data_mask_v,....     
%     'clim',[0 10],'colormap',jet,...
%     'threshold',1.0); %L-1�� R-2
% himg = imshow(rgbimg);%
% title('2back-1.0'); %L-1�� R-2


[rawimg,Lookup,rgbimg] = cvnlookupimages(ssname,val4,viewhemis,viewpt,Lookup,...
    'xyextent',[1 1],'surftype',surf_tpe,'surfsuffix',surfsuffix,...
    'text',upper(viewhemis),'rgbnan',10,...
    'roimask',data_mask_v,....     
    'clim',[0 0.3],'colormap',jet,...
    'threshold',0); %L-1�� R-2
figure;
himg = imshow(rgbimg);%
title('1back-1.0'); %L-1�� R-2

ori=cvnlookupviewpoint(ssname,hemi{hm},'occip',surf_tpe);
%色标卡
%figure; drawcolorbarcircular(cmapang,1);
%% draw roi
% labelname={'OFA'};
% labelname={'FFA-1'};
labelname={'FFA-2'};

R1=zeros(length(val4),1); %
L = Lookup;
%press Escape to erase and start again
%double click on final vertex to close polygon
%right click on first vertex, and click "Create mask" to view the result
%Keep going until user closes the window
for i = 1:length(labelname)
    R = [];
    while(ishandle(himg))
        [r,rx,ry]=roipoly();
        if(isempty(r))
            continue;
        end
        R=spherelookup_image2vert(r,L)>0;
        
        imgroi=spherelookup_vert2image(R,L,0);
        
        %quick way to merge rgbimg background with roi mask
        tmprgb=bsxfun(@times,rgbimg,.75*imgroi + .25);
        set(himg,'cdata',tmprgb);
        
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
    
    %%  draw another roi

        %% [~,L,rgbimg1]=cvnlookupimages(ssname,val4,hemi{hm},ori, ...
            %% 'cmap', cmapang(64), ...
             %%'roimask',betamask_result_a,...
            %% 'xyextent',[1 1],'surfsuffix','orig','circulartype',1,...
             %%'surftype',surf_tpe,'clim',[0 360],'threshold',0.7);
     %%figure;
    %% h=imshow(rgbimg1);
end
