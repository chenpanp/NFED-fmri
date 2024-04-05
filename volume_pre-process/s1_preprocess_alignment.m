%STEP 1
close all;
%% preparing for the data
% subjNO = {'SUBJ03'};
% load the anatomy
T1name = ['/home/xianchige/YZY/20220523data/sub01/sub01/mri/T1.nii.gz'];%T1路径，是原始数据转格式之后的T1，并非freesurfer重构出的T1像

vol1 = load_untouch_nii(gunziptemp(T1name));
vol1size = [1 1 1];  % 1mm isotropic 结构像分辨率
vol1 = double(vol1.img);
vol1(isnan(vol1)) = 0;
vol1 = fstoint(vol1);

% load the mean EPI
EPIname = ['/home/xianchige/YZY/20220523data/sub01/session07/preprocess/mean.nii'];%预处理运行出的结果

vol2 = load_untouch_nii(EPIname);
vol2size = vol2.hdr.dime.pixdim(2:4);
vol2 = double(vol2.img);
vol2(isnan(vol2)) = 0;

% keep a copy
vol1orig = vol1;
vol2orig = vol2;

% pre-condition the volumes
vol1 = preconditionvolume(vol1);
vol2 = preconditionvolume(vol2);

%STEP 2
%% start the alignment
% alignvolumedata(vol1,vol1size,vol2,vol2size,tr);
alignvolumedata(vol1,vol1size,vol2,vol2size);

%STEP 3
% TIP: 129 69 128   90 90 0      0 0 0   1 -1 -1   0 0 0
T=[129 69 128   90 90 0      0 0 0   1 -1 -1   0 0 0];
% T=[-0.300055801176444 2.97797264131109 0.226334167225528 35.2434672459151;-3.09728333963963 -0.319445724231673 0.245007856774599 218.25273133372;0.291001377614183 -0.212514085658706 2.9845778333177 91.6170035295175;0 0 0 1];
% go do some manual alignment (to get a reasonable starting point)
%tr =maketransformation([0 0 0],[1 2 3],[128.27046240153 105.422886515261 177.144882211424],[1 2 3],[2.2101523235715 2.04883338023553 -86.1432581327081],[110 110 56],[220 220 112],[1.01114774620826 -1.00682687199361 1.00243485765372],[0 0 0],[0.00155908822752155 0.0083997959605894 0.00477960388654554],[0 0 0]);
tr = alignvolumedata_exporttransformation;  % get a good seed and save just in case
% tr = maketransformation([0 0 0],[1 2 3],[129 69 128],[1 2 3],[90 90 0],[96 96 31],[192 192 62],[1 -1 -1],[0 0 0],[0 0 0],[0 0 0]);
%tr = maketransformation([0 0 0],[1 2 3],[124.891086780917 126.277491063275 160.112091574247],[1 2 3],[-3.00690283929811 5.97530318224449 -93.4549983643315],[96 96 60],[192 192 120],[1 -1 1],[0 0 0],[0 0 0],[0 0 0]);
%tr = maketransformation([0 0 0],[1 2 3],[127.5 119.5 176.5],[1 2 3],[0 0 0],[110 110 56],[220 220 123.200002670288],[-1 1 1],[0 0 0],[0.06 -0.04 0],[0 0 0]);
%sub01 tr = maketransformation([0 0 0],[1 2 3],[129.287717647195 108.300624757841 159.651069228947],[1 2 3],[0.000828408618643968 0.000653741521594938 0.000218431291548702],[110 110 56],[220 220 123.200002670288],[-0.998760143406941 1.00142922060333 1.00015249262722],[0 0 0],[0.000849517744470843 -0.000867091915647068 9.25454184115595e-05],[0 0 0]);
%%%T=[-1.99752028666934 -0.00170456418587189 -0.000159748908150836 240.249249713541;-7.61523693057366e-06 2.00285843451349 -0.00176847643015558 -2.80719413374915;2.27915906335119e-05 2.89776064042259e-05 2.20033550809746 96.9386340577337;0 0 0 1]; 
%STEP 4
% manually define ellipse to be used in the auto alignment
[f,mn,sd] = defineellipse3d(vol2);  % be very careful here. avoid crazy edges.
%sub05 mn = [0.508975207865301 0.46393266938973 0.0660068395703445]
%sd = [0.326420549637897 0.353959652776408 0.710176304448051]
%mn = [0.517375910871817 0.50706031177073 0.398182758209471]
%sd = [0.36254147566999 0.359045368171341 0.498563429520635]
%STEP 5
% auto-align (rigid-body, mutual information metric)
alignvolumedata_auto(mn,sd,[1 1 1 1 1 1 0 0 0 0 0 0],[1 1 1],[],[],[],1);


%% refine the fitting
% could try (affine):
alignvolumedata_auto(mn,sd,[0 0 0 0 0 0 1 1 1 1 1 1],[1 1 1],[],[],[],1);
alignvolumedata_auto(mn,sd,[1 1 1 1 1 1 0 0 0 0 0 0],[1 1 1],[],[],[],1);
alignvolumedata_auto(mn,sd,[0 0 0 0 0 0 1 1 1 1 1 1],[1 1 1],[],[],[],1);
alignvolumedata_auto(mn,sd,[1 1 1 1 1 1 0 0 0 0 0 0],[1 1 1],[],[],[],1);
alignvolumedata_auto(mn,sd,[0 0 0 0 0 0 1 1 1 1 1 1],[1 1 1],[],[],[],1);

alignvolumedata_auto(mn,sd,[1 1 1 1 1 1 0 0 0 0 0 0],[1 1 1],[],[],[],1);
alignvolumedata_auto(mn,sd,[0 0 0 0 0 0 1 1 1 1 1 1],[1 1 1],[],[],[],1);
alignvolumedata_auto(mn,sd,[1 1 1 1 1 1 0 0 0 0 0 0],[1 1 1],[],[],[],1);
alignvolumedata_auto(mn,sd,[0 0 0 0 0 0 1 1 1 1 1 1],[1 1 1],[],[],[],1);
alignvolumedata_auto(mn,sd,[1 1 1 1 1 1 0 0 0 0 0 0],[1 1 1],[],[],[],1);
alignvolumedata_auto(mn,sd,[0 0 0 0 0 0 1 1 1 1 1 1],[1 1 1],[],[],[],1);

%STEP 6
%% results output
% record transformation
tr = alignvolumedata_exporttransformation;
%xyf:tr = maketransformation([0 0 0],[1 2 3],[129.654299492701 129.116284268622 159.145457469417],[1 2 3],[-2.21876661627037 4.38468766701754 -91.7481302030786],[96 96 60],[192 192 120],[1.0004857830497 -1.00114010297265 1.00119477234223],[0 0 0],[-0.000133431940244175 0.0005162688911608 0.00037470811146875],[0 0 0]);
% what was it?
%tr = maketransformation([0 0 0],[1 2 3],[128.637378933136 69.9366199564417 135.521529150061],[1 2 3],[-5.04425263749522 -84.931871906197 94.6329721619243],[96 96 31],[192 192 62],[-1 1 -1],[0 0 0],[0 0 0],[0 0 0]);

% inspect the alignment
% savedir = ['/home/stone/litong/Desktop/litong/rawdata/fMRI/Alignment/'];
savedir = ['/home/xianchige/YZY/20220523data/sub01/session07/preprocess/alignment/'];%需自行新建alignment文件夹
%  savedir = ['/home/stone/litong/Desktop/litong/retinotopic/duanxiaohan2/fmri/Alignment/'];
matchvol = extractslices(vol1orig,vol1size,vol2orig,vol2size,tr);

imwrite(uint8(255*makeimagestack(rotatematrix(vol2orig,1,2,0),1)),gray(256),[savedir 'vol.png']);
imwrite(uint8(255*makeimagestack(rotatematrix(matchvol,1,2,0),1)),gray(256),[savedir 'matchvol.png']);

% convert the transformation to a matrix
T = transformationtomatrix(tr,0,vol1size);
fprintf('T=%s;\n',mat2str(T));

% what was it?
% T=[0.0142709254657683 -1.99989294167434 0.0149854104879037 224.700280151445;-0.176103138836984 0.0136710088715292 1.9921849281651 45.9396194091246;-1.99218072101356 -0.0155346502237758 -0.175996163209066 235.711663266417;0 0 0 1];

% save the results
cd(savedir)
savename = ['alignment.mat'];
save(savename,'tr','T','matchvol');
