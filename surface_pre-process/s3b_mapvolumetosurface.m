% change to data directory
% cd  /home/xianchige/TL/retinotopic/zhangchi;
clear all;
sub_dir = '/home/amax/YZY/20220523data/sub05/session08';
% define FreeSurfer stuff
fsdir = '/home/amax/YZY/20220523data/sub05/sub05';%皮层的位置
subjectid = 'sub05';
% define
hemis = {'lh' 'rh'}; 
interptype = 'linear';  % what type of interpolation?
outputname = 'surfacedata';    % use this in the output filename

%%%%%%%%%%%%%%%%%%%%%%%%%%

% load surfaces
vertices = {};
for p=1:length(hemis)
  vertices{p} = freesurfer_read_surf_kj(sprintf('%s/surf/%s.graymid',fsdir,hemis{p}));
  vertices{p} = bsxfun(@plus,vertices{p}',[128; 129; 128]);  % NOTICE THIS!!!%相加
  vertices{p}(4,:) = 1;  % now: 4 x V  加一行1
end

% load in the alignment information

a1 = load([sub_dir '/preprocess/alignment/alignment.mat']);

for i =1:6%prf实验
    % get the volume that we want to map
%     vol = load_untouch_nii(sprintf('loc/preprocess/run0%1d.nii',i));
    vol = load_untouch_nii(sprintf([sub_dir '/preprocess/run0%1d.nii'],i));
    vol = double(vol.img);

    %  figure;
    %  imagesc(makeimagestack(vol),[0 max(max(max(vol)))]);
    %  colormap(jet)
    
    % tseries = squish(vol,3);
    % loc的TR是2s，session1与session2的TR=1.88 需要插值到1s
    %vol = tseriesinterp(vol,1.88,1,4); %插值代码，参数1：4D提数据，参数2：原始TR;参数3：新的TR;参数4：新数据的维数（默认是2）
    % map volume onto surfaces
    for p=1:length(hemis)

      % get coordinates of the surfaces in volume space
      coord = volumetoslices(vertices{p},a1.tr);
      coord = coord(1:3,:);%删去最后一行1

      % perform interpolation to map the volume values onto the surfaces
      vals = ba_interp3_wrapper(vol,coord,interptype);
      vals_size = size(vals);
      vals=reshape(vals, vals_size(2:3));
      
      
      % write out the .mgz file to [lh,rh].<outputname>.mat
      save(sprintf([sub_dir '/preprocess/surfacedata/surface_run0%1d/%s.%s.mat'],i,hemis{p},outputname),'vals');
      % write out the .mgz file to [lh,rh].<outputname>.mgz
      cvnwritemgz(subjectid,outputname,vals,hemis{p},sprintf([sub_dir '/preprocess/surfacedata/surface_run0%1d'],i));
    end
    
end


