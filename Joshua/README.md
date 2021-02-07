# BAD2021

# A Note on SCVMC Angiograms
The SCMVMC provides Time Of Flight (TOF) angiograms in the form of jpeg images. Each jpeg image represents one frame of the TOF. The SCVMC also provides Maximum Intensity Projections (MIPs) in jpeg format.
However, the SCVMC stores these images in the same directory.
It is necessary to separate the TOFs from the MIPs.

# Usage of SCVMC Dataset Class
Set the root directory where "100" and "allseries" are located.
Pass the root directory to the dataset class constructor.
Set the stitch argument to true to stitch three MIPs together.
The dataset takes about ten minutes to process the data.

# RSNA 2019 Challenge
https://www.rsna.org/education/ai-resources-and-training/ai-image-challenge/rsna-intracranial-hemorrhage-detection-challenge-2019
