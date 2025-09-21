# GLMsingle


GLMsingle is a toolbox for obtaining accurate single-trial estimates in fMRI time-series data. We provide both MATLAB implementations. 

GLMsingle is detailed in the following paper:

**[Prince, J.S., Charest, I., Kurzawski, J.W., Pyles, J.A., Tarr, M., Kay, K.N. Improving the accuracy of single-trial fMRI response estimates using GLMsingle. *eLife* (2022).](https://doi.org/10.7554/eLife.77599)**

For additional documentation and FAQ on GLMsingle,
please see: **https://glmsingle.readthedocs.io**

For a lecture overview, implementation guide, and demo of GLMsingle,
please see: **https://cbmm.mit.edu/video/glmsingle-toolbox-improving-single-trial-fmri-response-estimates**

For a video walkthrough of the figure outputs from GLMsingle,
please see: **https://www.youtube.com/watch?v=aZFh-YUZUYE**

GLMsingle can be viewed as a wholesale replacement of its predecessor,
GLMdenoise (http://github.com/kendrickkay/GLMdenoise).

If you have questions or discussion points, please use the Discussions
feature of this github repository. If you find a bug, 
please let us know by raising a github Issue.

## MATLAB

To install: 

```bash
git clone --recurse-submodules https://github.com/cvnlab/GLMsingle.git
```

This will also clone [`fracridge`](https://github.com/nrdg/fracridge) as a submodule.

To use the GLMsingle toolbox, add it and `fracridge` to your MATLAB path by running the `setup.m` script.

## Python

To install: 

```bash
pip install git+https://github.com/cvnlab/GLMsingle.git
```

Running the example scripts requires:

- installing jupyter notebook or jupyter lab
- cloning the GLMsingle repository in order to get the example scripts located in `examples`:




## Example scripts

We provide a number of example scripts that demonstrate usage of GLMsingle. You can browse these example scripts here:



(MATLAB Example 1 - event-related design) https://htmlpreview.github.io/?https://github.com/kendrickkay/GLMsingle/blob/main/matlab/examples/example1preview/example1.html



If you would like to run these example scripts, the Python versions are available in `matlab/examples`, and the MATLAB versions are available in `/GLMsingle/matlab/examples`.

The first two notebooks contain a full walkthrough of the process of loading an example dataset and design matrix, estimating neural responses using GLMsingle, estimating the reliability of responses at each voxel, and comparing those achieved via GLMsingle to those achieved using a baseline GLM.




