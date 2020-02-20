!#/bin/bash


#Step 1. Reconstruct the volumetric image from the 2D data (TomoPy)
#  Inputs:
#  Outputs:
#  Command-line:

#Step 2. Noise removal (may involve multiple sub-stages)
#         (a) standard image processing methods (mean / median filter, non-local means, etc.)
#           Inputs:
#           Outputs:
#           Command-line:

#         (b) noise2noise
#           Inputs:
#           Outputs:
#           Command-line:

#Step 3. Segmentation to identify the phases
#  Inputs:
#  Outputs:
#  Command-line:

#Step 4. Quantitative analysis (e.g. extract surface area, analyze topology, etc.)
#  Inputs:
#  Outputs:
#  Command-line:

#This basic workflow could be customized to involve additional  steps, like
#running the data through a super-resolution network, or by performing direct
#simulation on the images. Depending on the particular information that is of
#interest, there are many possible analyses that can be performed once the data
#has been segmented.
#
#In the LBPM segmentation tool, both segmentation and analysis are performed
#together to avoid reading the data multiple times. However, this is not as
#flexible as allowing users to augment the workflow with their own python
#scripts, for example.

