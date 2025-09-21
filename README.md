# A fMRI dataset in response to large number of short naturalistic facial expression videos
Naturalistic facial expressions dataset (NFED),a large-scale dataset of whole-brain functional magnetic resonance imaging (fMRI) responses to 1,320 short (3s) facial expression video clips.NFED offers researchers fMRI data that enables them to investigate the neural mechanisms involved in processing emotional information communicated by facial expression videos in real-world environments.
The dataset contains raw data, pre-processed volume data,pre-processed  surface data and suface-based analyzed data.
To get more details, please refer to the paper at {website} and the dataset at https://openneuro.org/datasets/ds005047

## Preprocess procedure
The MRI data were preprocessed by using Kay et al, combining code written in MATLAB and certain tools from FreeSurfer, SPM,and FSL(http://github.com/kendrickkay). We used FreeSurfer software (http://surfer.nmr.mgh.harvard.edu) to construct the pial and white surfaces of participants from the T1 volume. Additionally, we established an intermediate gray matter surface between the pial and the white surfaces for all participants.

**code: ./volume_pre-process/**

Detailed usage notes are available in codes, please read carefully and modify variables to satisfy your customed environment.
## GLM of main experiment
We utilized a single-trial General Linear Model (GLMsingle) (https://github.com/cvnlab/GLMsingle) approach, an advanced denoising approach in MATLAB R2019a, to model the pre-processed fMRI data from main experiment. For single trials, the method of GLM was developed to offer estimations of BOLD response magnitudes ('betas'). GLMsingle requires only fMRI time series data and a design matrix as inputs, integrating three techniques to enhance the accuracy of experimental GLM beta estimates. Firstly, for each voxel, a custom HRF is identified from a library of candidate functions. Secondly, cross-validation is utilized to derive a set of noise regressors from voxels unrelated to the experimental paradigm. Thirdly, to improve the stability of beta estimates for closely spaced trials, ridge regression is employed on a voxel-wise basis to regularize the betas. In this study, three betas were calculated by analyzing the BOLD response corresponding to individual video onset ranging from 1 to 3 seconds with 1-second intervals. We produced individual GLMsingle models for each session (consisted of 4 training runs and 2 test runs). In general, for each video within the training set, 2 (repetitions) x 3 (seconds) betas were acquired. Similarly, for each video within the testing set, 10 (repetitions) x 3 (seconds) betas were acquired. The utilization of repetitions enabled us to acquire video-evoked responses with a high signal-to-noise ratio (SNR).
**code: ./GLMsingle-main-experiment/matlab/main.m**

#### retinotopic mapping

The fMRI data from the the population receptive field experiment were analyzed by a pRF model implemented in the analyzePRF toolbox (http://cvnlab.net/analyzePRF/) to characterize individual retinotopic representation. Make sure to download required software mentioned in the code.

**code: ./Functional-localizer-experiment-analysis/s4a_analysis_prf.m**

#### fLoc experiment

We used GLMdenoise,a data-driven denoising method,to analyze the pre-processed fMRI data from the fLoc experiment.We used a "condition-split" strategy to code the 10 stimulus categories, splitting the trials related to each category into individual conditions in each run. Six response estimates (beta values) for each category were produced by using six condition-splits.To quantify selectivity for various categories and domains,we computed t-values using the GLM beta values after fitting the GLM.The regions of interest with category selectivity for each participant were defined by using the resulting maps.
**code: ./Functional-localizer-experiment-analysis/s4a_analysis_floc.m**


## Validation
### Basic quality control
**code: ./validation/FD/FD.py**
**code: ./validation/tSNR/tSNR.py**
### noise celling
The code are available at  https://openneuro.org/datasets/ds005047."./validation/code/noise_celling/sub-xx" store the intermediate files required for running the program
**code: ./validation/noise_celling/Noise_Ceiling.py**

### Correspondence between human brain and DCNN
The code are available at  https://openneuro.org/datasets/ds005047. We combined the data from main experiment and functional localizer experiments to build an encoding model to replicate the hierarchical correspondences of representation between the brain and the DCNN. The encoding models were built to map artificial representations from each layer of the pre-trained VideoMAEv2 to neural representations from each area of the human visual cortex as defined in the multimodal parcellation atlas.
**code: ./validation/dnnbrain/**

### Semantic metadata of action and expression labels reveal that NFED can encode temporal and spatial stimuli features in the brain
The code are available at  https://openneuro.org/datasets/ds005047."./validation/code/semantic_metadata/xx_xx_semantic_metadata" store the intermediate files required for running the program.
**code: ./validation/semantic_metadata/**

## results
 The results can be viewed at  "https://openneuro.org/datasets/ds005047/derivatives/validation/results/brain_map_individual".

## Whole-brain mapping
The whole-brain data mapped to the cerebral cortex as obtained from the technical validation.
**code: ./show_results_allbrain/Showresults.m**

## Mannually prepared environment
We provide the *requirements.txt* to install python packages used in these codes. However, some packages like *GLM* and *pre-processing* require external dependecies and we have provided the packages in the corresponding file.

## stimuli
The video stimuli used in the NFED experiment are saved in the "stimuli_1" and "stimuli_2" folders.
