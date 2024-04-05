clear all;

load('/home/amax/YZY/code/retinotopysmall6.mat'); %视网膜刺激时间-模式文件
% stimulus is 200*200*300 200*200pix  300 means 300 seconds with TR=1

COpRF_options = [];
%run = 1
TR = 1;
% tseries ={};
data={};



%% Load in preprocessed data

data_dir= '/home/amax/YZY/20220523data/sub01/session01/preprocess/result_for_prf_mean';%存放prf分析结果的路径
data_dir2 = '/home/amax/YZY/20220523data/sub01/session01/preprocess/surfacedata';%surface data dir
runix=[2 4];%run number
hemis = {'lh','rh'};
data = {};
for p=1:numel(runix)
    %GLMdenoise wants data in (vertices)x(time)
    % GLM want bad data, so use all data  
    %convert to single() to save memory
    data{p} = [];
    for hm = 1:2 
         load(sprintf([data_dir2 '/surface_run0%1d/%s.surfacedata.mat'],runix(p),hemis{hm}));
         vol = single(vals);
         %在这平均
         data{p} = [data{p}; vol(:,1:300)]; %插值后的数据 TR=1s 一共是300s
         
    end
end
data_sum=data{1}+data{2};
data_mean={};
data_mean{1}=data_sum/2;
stimulus{1} = stim;

%% setup the parameters and perform the pRF analysis
% pRF_options = struct('display','off');  % in this mode, we obtain correlation (r) values
pRF_options = [];
% perform the analysis
results = Convex_pRF_fit_parallel(stimulus,data_mean,TR,pRF_options);

% [results, B, G] = Convex_pRF_fit_parallel(stimulus,data,TR,COpRF_options);

savedir = [data_dir];
savename = sprintf(['/retino_pRF_run_mean.mat']) ;
%savename = sprintf(['/retino_pRF_run0%1d'],runix(p)) ;
% cd(savedir) 
save([savedir savename], 'results');

