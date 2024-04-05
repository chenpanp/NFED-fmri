# A large-scale fMRI dataset in response to short naturalistic facial expressions videos

#FreeSurfer reconstruction
## 1.Setting up the FreeSurfer environment

export FREESURFER_HOME=~/soft/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.sh
export SUBJECTS_DIR=/home/xianchige/YZY/
tcsh
setenv FREESURFER_HOME /home/xianchige/soft/freesurfer
source $FREESURFER_HOME/SetUpFreeSurfer.csh

## 2.FreeSurfer reconstruction

recon-all -s sub01 -i /home/xianchige/YZY/20220523data/sub01/T1/DICOM_t1_mprage_sag_p2_iso_1mm_20220414161101_18.nii.gz  -all -openmp 32 -norandomness > /home/xianchige/YZY/20220523data/sub01/T1/reconlog.txt

##3.Converting T1 to NIFTI

cd /home/xianchige/YZY/20220523data/sub01/sub01
mri_convert mri/T1.mgz mri/T1.nii.gz

# volume to surface
##1. make mid-gray surfaces.
cd /home/xianchige/YZY/20220523data/sub02/sub02/surf/
mris_expand -thickness rh.white 0.5 rh.graymid
mris_expand -thickness lh.white 0.5 lh.graymid

##2.We run run the program(s3b_mapvolumetosurface.m)  for surface-based preprocessing