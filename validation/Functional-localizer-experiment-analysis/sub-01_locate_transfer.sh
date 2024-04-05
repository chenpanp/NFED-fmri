#where fsaverage and litong_dxh live
export SUBJECTS_DIR=/home/xianchige/YZY/20220523data/sub-01/
mri_surf2surf \
--hemi lh \
--srcsubject fsaverage \
--trgsubject sub-01 \
--sval fsaverageKastner2015Labels-LH.mgz \
--tval sub-01_lh.Kastner2015Labels.mgz

mri_surf2surf \
--hemi rh \
--srcsubject fsaverage \
--trgsubject sub-01 \
--sval fsaverageKastner2015Labels-RH.mgz \
--tval sub-01_rh.Kastner2015Labels.mgz


