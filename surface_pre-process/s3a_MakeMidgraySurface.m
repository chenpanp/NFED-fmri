% make mid-gray surface
% before we push volume onto surfaces,
% we need to make mid-gray surfaces,
% which is necessary in the next step.


% cd('/home/stone-ext1/freesurfer/subjects/litong_dxh1/surf/');
%cd('/home/xianchige/YQZD/subject/sub17/surf/');
cd('/home/xianchige/DPHZD/subject/sub09/surf/');

unix('mris_expand -thickness lh.white 0.5 lh.graymid');
unix('mris_expand -thickness rh.white 0.5 rh.graymid');

% for i = 1:length(subjid)  
%     cd(sprintf('/stone/ext1/freesurfer/subjects//%s/surf',subjid{i}));
%     %cd(sprintf('/stone/ext1/freesurfer/subjects/%s/surf',subjid{i}));
%     unix('mris_expand -thickness lh.white 0.5 lh.graymid');
%     unix('mris_expand -thickness rh.white 0.5 rh.graymid');
% end
