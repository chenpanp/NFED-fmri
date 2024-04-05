% start parallel MATLAB to speed up execution.
delete(gcp);
numworkers = 1;  % keep low to prevent memory swapping!?
if parpool('local')==0
   parpool('local',numworkers);
end


  datadir = '/home/xianchige/YZY/20220523data/sub05/session01';   %设置数据目录
  figuredir = [datadir '/preprocessfigures'];   %设置结果输出路径

  %%% CASE 2: separate DICOM directories [if you use this case, please comment out CASE 1 and CASE 3]

  % the first set of DICOMs are the fieldmaps; the second set of DICOMs are the magnitude brains.
  % after loading in the second set of DICOMs, we consider only the first N slices, where N is
  % the number of slices in the first set of DICOMs (the reason is that there may be two volumes
  % in the second set of DICOMs).
  
      fieldmapfiles = [];
      fieldmapfiles = cellfun(@(x,y) {x y}, ...
      matchfiles([datadir '/PH_field_mapping*']), ...   %filed mapping 相位像
      matchfiles([datadir '/MA_field_mapping*']), ...   %filed mapping 幅度像
      'UniformOutput',0);  
%       fieldmapfiles = cellfun(@(x,y) {x y}, ...
%       matchfiles([datadir '/field_mapping/213702*']), ...
%       matchfiles([datadir '/field_mapping/213701*']), ...
%       'UniformOutput',0);  

  
    %           cellfun(@(x,y) {x y}, ...
    %     matchfiles([datadir '/dicom/PH*field_mapping*']), ...
    %     matchfiles([datadir '/dicom/MR*field_mapping*']), ...
    %     'UniformOutput',0);

    % %%%% OOPS.  THE FIELDMAPS HAD TOO MANY SLICES IN THIS SESSION (ESSA MISTAKE)
    % %%%% SO WE USE THIS HACK:
    % fieldmapslicerange = 8:42-7;
  
% if you didn't acquire fieldmaps with the same slice thickness as the 
% functionals, we can work around this problem if your fieldmaps are
% a positive integer multiple of the slice thickness of the functionals,
% and if the total field-of-view in the slice dimension is the same.
% all we do is upsample the fieldmaps using nearest neighbor interpolation.
% this is done immediately and we then act as if the fieldmaps were acquired
% at the correct slice thickness.  (of course, we could be more flexible
% and fix other circumstances, but we'll do this as the need arises.)
% if you want the work around, supply the appropriate positive integer 
% for <fieldmapslicefactor>.  if [], do nothing special.
fieldmapslicefactor = 1;    %fieldmaping的层厚是功能像的几倍

% what are the time values to associate with the fieldmaps?
% if [], default to 1:N where N is the number of fieldmaps.
fieldmaptimes = [1 5 9];    %fieldmaping 在session中的runnumber，[1 5 9]也就是说先扫了场图然后3个bold又一个场图又3个bold又场图
% what is the difference in TE (in milliseconds) for the two volumes in the fieldmaps?
fieldmapdeltate = 7.22-4.76;    %TE2-TE1

% should we attempt to unwrap the fieldmaps? (note that 1 defaults to a fast, 2D-based strategy; 
% see preprocessfmri.m for details.)  if accuracy is really important to you and the 2D strategy 
% does not produce good results, consider switching to a full 3D strategy like 
% fieldmapunwrap = '-f -t 0' (however, execution time may be very long).
fieldmapunwrap = 1;%打开fieldmaps的方式，1 defaults to a fast, 2D-based strategy

% how much smoothing (in millimeters) along each dimension should we use for the fieldmaps?
% the optimal amount will depend on what part of the brain you care about.
% I have found that 7.5 mm may be a good general setting.
fieldmapsmoothing = [5 5 5];    %[7.5 7.5 7.5];  %[5 5 5]; 带宽沿每个维度必须大于沿该维度的fieldmap的体素大小
%（这是因为我们需要沿每个维度有多个数据点才能执行局部线性回归
 
% what DICOM directories should we interpret as in-plane runs?
% it is okay if you also match things that are not DICOM directories; we'll just ignore them.
inplanefilenames = [];     %matchfiles([datadir '/dicom/*Inplane*']);

% what DICOM directories should we interpret as EPI runs?
% it is okay if you also match things that are not DICOM directories; we'll just ignore them.
%epifilenames = matchfiles([datadir '/volumn_*']);
epifilenames = matchfiles([datadir '/volumn_*']);
% epifilenames = matchfiles([datadir]);   
%%isok = @(x) isempty(regexp(x,'SBRef'));
    %%epifilenames = epifilenames(cellfun(isok,epifilenames));
    %%epifilenames = epifilenames([1 4 5 6]);  % had to ignore some crap runs

% do you want the special "push alternative data" mode?  if so, specify the 'record.mat'
% file from the previous call and <mcmask> must not be {}.  otherwise, leave as [].
wantpushalt = [];
%%wantpushalt = [datadir '/preprocessfigures/record.mat'];

% this input is important only if you acquired something other than just the magnitude images
% for the EPI data.  the format is documented in dicomloaddir.m (the input <phasemode>).
% for the purposes of this script, the second element of <epiphasemode> must have exactly one
% element (and that element should probably be either 1 or 5).  if you acquired just the 
% magnitude images, just leave epiphasemode set to [].
epiphasemode = [];

% what is the desired in-plane matrix size for the EPI data?
% this is useful for downsampling your data (in order to save memory) 
% in the case that the data were reconstructed at too high a resolution.  
% for example, if your original in-plane matrix size was 70 x 70, the 
% images might be reconstructed at 128 x 128, in which case you could 
% pass in [70 70].  what we do is to immediately downsample each slice
% using lanczos3 interpolation.  if [] or not supplied, we do nothing special.
epidesiredinplanesize = [];%[ab]，其中A和B分别是平面内频率编码和相位编码矩阵大小。例如，[76 64]表示76个频率编码步骤和64个相位编码步骤。这种输入之所以必要，是因为<epis>的维度可能大于

%由于重建过程中的零填充，实际测量的矩阵大小。可以是[]，在这种情况下，我们默认为EPI数据的前两个维度的大小。


% what is the slice order for the EPI runs?
% special case is [] which means to omit slice time correction.

% episliceorder = 'interleavedalt';

  %   % HOW TO FIGURE OUT THE SLICE PARAMETER
  a = dicominfo('1.IMA');%ducanshu cengjianju spm之类的参数  读取相关参数，任意一个任务实验的图像
  mbfactor = 4; %multi-band factor :加速参数
  [d,ix] = sort(a.Private_0019_1029(1:end/mbfactor));%Private_0019_1029相位编码的时间？
  ord = [];
  for p=1:length(ix)
    ord(ix(p)) = p;
  end
  ord = repmat(ord,[1 mbfactor]);
  mat2str(ord);
  episliceorder = {};
  episliceorder{1} = ord;   %slice order，上面的代码都是求slice order的
  episliceorder{2} = 1; %TR？
% HOW TO FIGURE OUT THE SLICE PARAMETER
  % a = dicominfo('MR-ST001-SE007-0077.dcm');
  % mbfactor = 3;
  % [d,ix] = sort(a.Private_0019_1029(1:end/mbfactor));
  % ord = [];
  % for p=1:length(ix)
  %   ord(ix(p)) = p;
  % end
%   ord = repmat(ord,[1 mbfactor]);
%   mat2str(ord)


  
  % KEITH'S WAY:
%   dcmfile = 'MR-ST001-SE040-0033.dcm';
%   D=dicominfo(dcmfile);
%   slicetimes=D.Private_0019_1029;
%   slicediff=diff(sort(slicetimes));
%   slicediff=median(slicediff(slicediff>0));
%   sliceindex=(round(slicetimes/slicediff)+1)';
%   mat2str(sliceindex);

   
% what is the phase-encode direction for the EPI runs? (see preprocessfmri.m for details.)
% (note that 'InPlanePhaseEncodingDirection' in the dicominfo of the EPI files gives either COL 
% which means the phase-encode direction is up-down in the images (which is 1 or -1 in our
% convention) or ROW which means the direction is left-right in the images (which is 2 or -2 in
% our convention).  the problem is that we don't know if the phase direction has been flipped,
% which you can do manually via CV vars.  it appears that if you don't explicitly flip, you should
% use the positive version (i.e. 1 or 2), meaning that the direction is towards the bottom
% (in the case of 1) or to the right (in the case of 2).  but in any case you should always
% check the sanity of the results!
%epiphasedir = repmat([-1 1],[1 4]);
%epiphasedir = repmat([-1 -1 -1  1 1 1],[1 2]);

epiphasedir = 1;%phase-encode direction1表示相位编码方向沿第一矩阵维度定向（例如1->64）

% what is the total readout time in milliseconds for an EPI slice?
% (note that 'Private_0043_102c' in the dicominfo of the EPI files gives the time per phase-encode line in microseconds.
% I confirmed that this time is correct by checking against the waveforms displayed by plotter.)
  %epireadouttime = 0.999974 * (162*6/8/3);  % divide by 2 if you are using 2x acceleration
% OOPS, we need to remove partial fourier factor
% epireadouttime = 0.9999 * (66/3);  % divide by 2 if you are using 2x acceleratio n 
  % 162 is the EPI factor
  
epireadouttime = 0.68 * (110/2);
%0.72 echo spacing回波间隔时间
%96 epi factor
% 2 Accel factor PE
% what fieldmap should be used for each EPI run? ([] indicates default behavior, which is to attempt
% to match fieldmaps to EPI runs 1-to-1, or if there is only one fieldmap, apply that fieldmap
% to all EPI runs, or if there is one more fieldmap than EPI runs, interpolate each successive
% pair of fieldmaps; see preprocessfmri.m for details.)
%epifieldmapasst = {1 [1 2] [2 3] 3};    %%[splitmatrix([1:12; 2:13]',1) {4 4 4 4}];
%epifieldmapasst = {1 1 [1 2] [2 3]  [3 4] [4 5]  [5 6] [6 7]};
%%%%%%%epifieldmapasst = [];   %splitmatrix([1:12; 2:13]',1);
%每次EPI运行应使用什么fieldmap？
epifieldmapasst = splitmatrix([1:8; 2:9]',1)%[1 2] [2 3]  [3 4] [4 5]  [5 6] [6 7]插值取值

% how many volumes should we ignore at the beginning of each EPI run?
numepiignore = 0;

% what volume should we use as reference in motion correction? ([] indicates default behavior which is
% to use the first volume of the first run; see preprocessfmri.m for details.  set to NaN if you
% want to omit motion correction.)
motionreference = [1 1];  %[1 137]; %[1 137]; the index of the EPI run and h indicates the index of the volume within that run (after ignoring volumes according to <numepiignore>)
%motion correction should be performed. default: [1 1].

% for which volumes should we ignore the motion parameter estimates?  this should be a cell vector
% of the same length as the number of runs.  each element should be a vector of indices, referring
% to the volumes (after dropping volumes according to <numepiignore>).  can also be a single vector
% of indices, in which case we use that for all runs.  for volumes for which we ignore the motion
% parameter estimates, we automatically inherit the motion parameter estimates of the closest
% volumes (if there is a tie, we just take the mean).  [] indicates default behavior which is to 
% do nothing special.
epiignoremcvol = [];

% by default, we tend to use double format for computation.  but if memory is an issue,
% you can try setting <dformat> to 'single', and this may reduce memory usage.
dformat = 'single';%减少内存使用

% apply Gaussian spatial smoothing? 3-element vector indicating FWHM in mm.
% [] means do nothing.
epismoothfwhm = [];  % [3 3 3]平滑 如果提供，我们在切片时间校正后立即平滑EPIvolumes

% what cut-off frequency should we use for filtering motion parameter estimates? ([] indicates default behavior
% which is to low-pass filter at 1/90 Hz; see preprocessfmri.m for details.)
%motioncutoff = [];
motioncutoff = Inf;%用于运动参数估计的低通滤波器截止频率，只有在执行运动校正时，此输入才起作用。

% what extra transformation should we use in the final resampling step? ([] indicates do not perform an extra transformation.)
% tr = maketransformation([0 0 0],[1 2 3],[117.040207900659 115.623739214828 43.7254395779036],[1 2 3],[-0.670461404521411 0.704835952293662 -0.249037864439899],[116 116 42],[232 232 84],[1 1 1],[0 0 0],[0 0 0],[0 0 0]);
% extratrans = transformationtomatrix(tr,0,[2 2 2]);
extratrans = [];%将EPI volume的矩阵空间中的点映射到新位置。如果提供，则将在新位置对体积进行重新采样。

% what is the desired resolution for the resampled volumes? ([] indicates to just use the original EPI resolution.)
targetres = [];%重采样体积使用原始EPI分辨率

% should we perform slice shifting?  if so, specify band-pass filtering cutoffs in Hz, like [1/360 1/20].
% probably should be left as [] which means to do nothing special.
sliceshiftband = [];%将根据[低-高]对质心计算进行带通滤波，从而执行切片移位校正。

% these are constants that are used in fmriquality.m.  it is probably 
% fine to leave this as [], which means to use default values.
% NaN means to skip the fmriquality calculations.
fmriqualityparams = [];

% what kind of time interpolation should we use on the fieldmaps (if applicable)?
% ([] indicates to use the default, which is cubic interpolation.)
fieldmaptimeinterp = 'linear';%fieldmaps上使用什么样的时间插值,表示要用于<epifieldmapasst>中[G H]情况的插值类型。

% should we use a binary 3D ellipse mask in the motion parameter estimation?
% if [], do nothing special (i.e. do not use a mask).
% if {}, then we will prompt the user to interactively determine the
%   3D ellipse mask (see defineellipse3d.m for details).  upon completion,
%   the parameters will be reported to the command window so that you can
%   simply supply those parameters if you run again (so as to avoid user interaction).
% if {MN SD}, then these will be the parameters that determine the mask to be used.
%  mcmask = { [0.56 0.5 0.4] [0.32 0.26 0.4];};
 mcmask = {};%配准后生成的数据，可记录下来用于下次的初始化
%mcmask = { [0.52 0.52 0.0999999999999998] [0.3 0.32 0.74];};

% how should we handle voxels that have NaN values after preprocessing?
% if [], we use the default behavior which is to zero out all voxels that have a NaN
% value at any point in the EPI data.  see preprocessfmri.m for other options.
maskoutnans = [];%将预处理后具有NaN的所有体素归零

% savefile:  what .nii files (accepting a 1-indexed integer) should we save the final EPI data to?
%            (in the special EPI flattening case, we save the data to raw binary files (time x voxels) instead of .nii files.)
% savefileB: what .nii file should we save the valid voxels (binary mask) to?  ([] means do not save.)
% savefileC: what .nii file should we save the mean volume to?  ([] means do not save.)
% savefileD: what .nii file should we save the mad volume to?  ([] means do not save.)
% savefileE: what .nii file should we save the tsnr volume to?  ([] means do not save.)
% (we automatically make parent directories if necessary.)
savefile = [datadir '/preprocess/run%02d.nii'];
savefileB = [datadir '/preprocess/valid.nii'];
savefileC = [datadir '/preprocess/mean.nii'];
savefileD = [datadir '/preprocess/mad.nii'];
savefileE = [datadir '/preprocess/tsnr.nii'];

% what .txt file should we keep a diary in?
diaryfile = [datadir '/preprocess/diary.txt'];

%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DO NOT EDIT BELOW:

  mkdirquiet(stripfile(diaryfile));
  diary(diaryfile);
 preprocessfmri_standard;
  diary off;
