# System
import os
import sys

# Math
import numpy as np

# Image Handling
from skimage.io import imread
from skimage.color import rgb2gray

# File Handling
import pandas as pd

# Deep Learning API
import torch
from torch.utils.data import Dataset

# Angiogram Class
class Angiogram():
    def __init__(self, scan = None, diagnosis = None, name = None):
        self.scan = scan # Array or Tensor
        self.diagnosis = diagnosis # Integer
        self.name = name # String

    def set_scan(self, scan):
        self.scan = scan

    def set_diagnosis(self, diagnosis):
        self.diagnosis = diagnosis

    def set_name(self, name):
        self.name = name

    def get_scan(self):
        return self.scan

    def get_diagnosis(self):
        return self.diagnosis

    def get_name(self):
        return self.name

# Abstract Parent Class
class MyDataset(Dataset):
    @property
    def raw_dirs(self):
        raise NotImplementedError

    @property
    def processed_dirs(self):
        raise NotImplementedError

    def is_processed(self):
        raise NotImplementedError

    # Data processing occurs in this method
    def process(self):
        raise NotImplementedError

    # Data loading occurs in this method
    def load(self):
        raise NotImplementedError

    def __init__(self, root):
        super().__init__()
        self.root = root
        # Data processing occurs in this method
        if(not self.is_processed()):
            self.process()
        # Data loading occurs in this method
        self.load()

class SCVMC(MyDataset):
    @property
    def raw_dirs(self):
        # Normal and Abnormal cases in 100 and allseries, respectively
        return ["100", "allseries"]

    @property
    def processed_dirs(self):
        return [directory + "_processed" for directory in self.raw_dirs]

    # Check if the data is already processed
    def is_processed(self):
        is_processed = True
        for i in range(len(self.processed_dirs)):
            if(not os.path.exists(os.path.join(self.root, self.processed_dirs[i]))):
                is_processed = False
        return is_processed

    def process(self):
        print("Processing...")
        # Create a dataframe, in which we will write all subject identifiers, filepaths, and diagnoses
        # The dataframe will allow us to load the processed data quickly
        columns = ["Subject", "Filepath", "Diagnosis"]
        df = pd.DataFrame(columns = columns)

        # Iterate through raw data
        for i in range(len(self.raw_dirs)):
            # Make directory to store processed data
            os.mkdir(os.path.join(self.root, self.processed_dirs[i]))
            for subject in os.listdir(os.path.join(self.root, self.raw_dirs[i])):
                # List to compile frames of subject scan
                subject_scan = []
                # If there are Time of Flight images
                if(len(os.listdir(os.path.join(self.root, self.raw_dirs[i], subject, "tofs")))):
                    # Make a directory for the subject
                    os.mkdir(os.path.join(self.root, self.processed_dirs[i], subject))
                    # Load and append each frame from the Time of Flight images to the list
                    for frame in os.listdir(os.path.join(self.root, self.raw_dirs[i], subject, "tofs")):
                        subject_scan.append(rgb2gray(imread(os.path.join(self.root, self.raw_dirs[i], subject, "tofs", frame))))
                    # Process the scan
                    subject_scan_np = np.array(subject_scan, dtype = np.float64)
                    # Stitch Maximum Intensity Projections along each axis together
                    if(self.stitch):
                        stitched = np.concatenate([np.max(subject_scan_np, axis = 0), np.max(subject_scan_np, axis = 1), np.max(subject_scan_np, axis = 2)], axis = 0)
                        subject_scan_pt = torch.FloatTensor(stitched)
                    else:
                        subject_scan_pt = torch.FloatTensor(subject_scan_np)
                    # Save processed scan to corresponding directory
                    torch.save(subject_scan_pt, os.path.join(self.root, self.processed_dirs[i], subject, "tofs.pt"))
                    # Add processed scan to dataframe
                    df = df.append({"Subject" : subject, "Filepath" : os.path.join(self.root, self.processed_dirs[i], subject, "tofs.pt"), "Diagnosis" : i}, ignore_index = True)

        # Reset dataframe indices
        df.reset_index(drop = True, inplace = True)
        # Save dataframe for loading
        df.to_csv(os.path.join(self.root, "fout.txt"))
        print("Complete!")

    def load(self):
        print("Loading...")
        self.meta = pd.read_csv(os.path.join(self.root, "fout.txt"))
        print("Complete")

    def __init__(self, root, stitch = False):
        # Called in the process() method
        self.stitch = stitch
        # Called in the load() method
        self.meta = None
        # The process() method is called in the super class initialization
        super().__init__(root)
        

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        angiogram = Angiogram()
        angiogram.set_scan(torch.load(self.meta.at[idx, "Filepath"]))
        angiogram.set_diagnosis(self.meta.at[idx, "Diagnosis"])
        angiogram.set_name(self.meta.at[idx, "Subject"])
        return angiogram