%% Initialize some file locations
clear all;
%close all;
ssname='sub07';
data_dir='/home/amax/YZY/20220523data/sub07/session01/surfacedata';

%% 1back_GLM
outpath='/home/amax/YZY/20220523data/sub07/session01/GLManalysis/1back';

%% Set surface drawing function for GLMdenoisedata

% generally, this section is a copy-and-paste job

hemis = {'lh','rh'};
surfsuffix='orig'; %Non-dense surfaces (standard freesurfer outputs); if dense, would be 'DENSETRUNCpt'

% Create empty value struct: struct with .numlh, .numrh, .data=(numlh+numrh)x1
valstruct=valstruct_create(ssname,surfsuffix);

[viewpt, ~, viewhemis] = cvnlookupviewpoint(ssname,hemis,'occip','sphere');

[~ , Lookup] = cvnlookupimages(ssname,valstruct,viewhemis,viewpt,[],'xyextent',[1 1],'surfsuffix',surfsuffix);
cathemis = @(v)spherelookup_vert2image(struct('numlh',valstruct.numlh,'numrh',valstruct.numrh,'data',v(:,1)),Lookup,nan);

% set up GLMdenoise options
clear opt;
opt.drawfunction = @(vals)cathemis(vals(:));

%% Load in preprocessed data

% define
runix=[3 4];   % in this session%yang
% load
data = {};
for p=1:numel(runix)
    %GLMdenoise wants data in (vertices)x(time)
    % GLM want bad data, so use all data  
    %convert to single() to save memory   
    data{p} = [];
    for hm = 1:2 
         load(sprintf([data_dir '/surface_run0%1d/%s.surfacedata.mat'],runix(p),hemis{hm}));
         vol = single(vals);
         data{p} = [data{p}; vol]; 
    end
%      FOR THIS EXPERIMENT SPECIFICALLY: trim first 9 timepoints from each scan
%      Often: remove last timepoint since it is extrapolated during preproc resampling
%      实际采集数据156TR，前6个TR对应的是倒计时，后面的150个TR，跟MAT文件对应
       data{p}=data{p}(:,18:end-2); % 丢弃1：7和最后一个数据点，则对应MAT文件的第2-149 %yang discard 1-15vol,12s countdown
%         for j=1:length(labelname)
%             % label_vol = read_label(ssname, sprintf('%s.%s',hemis{hm},labelname{j}));
%         end
end
     
timepoints = size(data{p});
num_timepoints = timepoints(2);
num_conditions  = 3; % 5 conditions; 5 types images
     
%% Create design matrix   clear

load('/home/xianchige/YZY/20220523data/sub07/sub21_floc.mat');
stim_label=zeros(390,numel(runix));
stim_label(16:390,:) = back_1(:,1:2);%same design for each run,fist 6TR is countdown(12 seconds)%yang_back1:2 shi 2 ge run
%delete lo_label;      
design={};
for p=1:numel(runix)
         %GLMdenoise wants (time)x(conditions)
         %ie: dim1 of design must match dim2 of data 
          design{p}=zeros(num_timepoints,num_conditions);     
          
          %GLMdenoise just wants a single "1" at the START of each stimulus
          %event, regardless of how long it lasted. We will handle the stimulus
          %duration later.         
          %totallly 1 conditions, image stimuli or not
          for i = 4:5:(390-19)  % trails each run 只用把每个trial的第一个设为1就行，因为后面设置了trial的时长
              if((stim_label(i+17,p)==2)||(stim_label(i+17,p)==5))
                  design{p}(i,1)=1;  % condition1 face images
              end
              if((stim_label(i+17,p)==3)||(stim_label(i+17,p)==6))
                  design{p}(i,1)=1;  % condition1 face images
              end
              if((stim_label(i+17,p)==1)||(stim_label(i+17,p)==7))
                  design{p}(i,2)=1;  % condition2 scene images
              end
              if((stim_label(i+17,p)==4)||(stim_label(i+17,p)==8))
                  design{p}(i,2)=1;  % condition2 scene images
              end
              if((stim_label(i+17,p)==9)||(stim_label(i+17,p)==10))
                  design{p}(i,3)=1; % condition3 car images
              end
          end
end
     
     
     %% run GLMdenoise

        % some more setup
        % opt.numboots = 0; %no bootstrapping for this 2-run, single condition experiment
        opt.wantparametric=1; %use parametric to estimate noise instead
        % the following are used to generate an HRF
        stimdur=4; %Each "1" in design, is the start of an 8*2 = 16 second event
        tr=0.8; %TR (in seconds), used for constructing an HRF for 18 second stimulus

        % call GLMdenoise
        % Note: ...,'assume',[],... = use canonical HRF (don't try to optimize)
        results = GLMdenoisedata(design,data,stimdur,tr,'optimize',[],opt,sprintf('%s/GLMfigures',outpath));

        % when you actually run, you should save your results:
          %note the '-struct' argument.  This makes it easier when you want to load
          %in all the elements of the structure later....
        % save([outpath '/results.mat'],'-struct','results');
        %
        % NOTE: the results are in /stone/ext4/howtos/glmdenoise_surface_data/results.mat
        %
        % note that the results might be large. if you use the bootstrap option,
        % consider removing the individual bootstrap results before saving
        % (contained in results.models).
        save([outpath '/results.mat'],'-struct','results');
           

        


% % %% inspect outputs
% % 
% % % results.parametric.designmatrix contains the final design matrix
% % % results.parametric.parameters contains the raw beta weights (not converted to PSC)
% % % results.parametric.parametersse contains standard error estimates on those beta weights
% % 
% % % results.R2 is the non-cross-validated R2 (e.g. "FinalModel.png")
% % % results.pcR2final is the cross-validated R2 when using the selected number of PCs (e.g. PCcrossvalidation13.png)
% % 
% % % results.modelmd{2} has the beta weights (estimated using GLMdenoise's method (not parametric))
% % 
% % look at quasi-t-values for color selectivity
% tval = results.parametric.parameters(:,1) ./ results.parametric.parametersse(:,1);
% figure;hist(tval(:),100);
% % look at some time-series data
% goodix = find(results.R2>40);
% figure; hold on;
% plot([data{1}(goodix,:) data{2}(goodix,:)]');
% plot(results.parametric.designmatrix(:,1)*200+300,'k-','LineWidth',5);

