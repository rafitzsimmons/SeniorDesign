# System
import os
import sys

# Math and Plotting
import numpy as np
import matplotlib.pyplot as plt

# Deep Learning
import torch
from torch.nn.functional import softmax

# Metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

# Trainer Class to Wrap Model
class Trainer():
    def __init__(self, model, loss, optim):
        # Model
        self.model = model
        self.loss = loss
        self.optim = optim
        
        # For file outputs, such as loss.txt, senistivity, specificity, ROC, etc.
        self.fout = "fout"
        if(not os.path.exists(self.fout)):
            os.mkdir(self.fout)

    # Modular training of one epoch
    def train_step(self, train_loader):
        if(self.model.training):
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            loss = 0.0
            for batch_idx, batch_data in enumerate(train_loader):
                data, batch_labels = batch_data
                data = data.to(device)
                batch_labels = batch_labels.to(device)
                self.optim.zero_grad()
                batch_probs = softmax(self.model(data), dim = -1)
                batch_probs = torch.squeeze(batch_probs, dim = 1)
                batch_probs = torch.squeeze(batch_probs, dim = 1)
                batch_loss = self.loss(batch_probs, batch_labels)
                batch_loss.backward()
                self.optim.step()
                loss += batch_loss
            return loss/len(train_loader)
        else:
            sys.error("Set model.training to True.")
    
    # Modular testing of validation set
    def test_step(self, test_loader):
        if(not self.model.training):
            # Omitting gradient calculations improves testing speed
            with torch.no_grad():
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                labels = []
                max_probs = []
                preds = []
                for batch_idx, batch_data in enumerate(test_loader):
                    data, batch_labels = batch_data
                    data = data.to(device)
                    batch_labels = batch_labels.to(device)
                    batch_probs = softmax(self.model(data), dim = -1)
                    batch_probs = torch.squeeze(batch_probs, dim = 1)
                    batch_probs = torch.squeeze(batch_probs, dim = 1)
                    batch_max_probs, batch_preds = torch.max(batch_probs, dim = -1)
                    for label in batch_labels:
                        labels.append(label.cpu().detach().numpy().item())
                    for max_prob in batch_max_probs:
                        max_probs.append(max_prob.cpu().detach().numpy())
                    for pred in batch_preds:
                        preds.append(pred.cpu().detach().numpy().item())
                tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
                fpr, tpr, _ = roc_curve(labels, max_probs)
                auc = roc_auc_score(labels, max_probs)
                return tn, fp, fn, tp, tpr, fpr, auc
        else:
            sys.error("Set model.training to False.")
            
    def train(self, train_loader, epochs, verbose = False):
        self.model.training = True
        
        with open(os.path.join(self.fout, "loss.txt")) as fp:
            for i in range(epochs):
                epoch_loss = self.train_step(train_loader)
                # Dump loss to comma separated file output
                fp.write(str(epoch_loss) + ',')
                if(verbose):
                    print("Epoch: {:d}\tLoss: {:f}".format(i, epoch_loss))
            # Read file output
            contents = fp.read()
        
        # Extract contents
        contents = contents.split(',')
        contents = contents.pop() # The last item is empty
        for i in range(len(contents))
            contents[i] = np.float(contents[i])
        loss = np.array(contents)

        # Plot loss
        fig, ax = plt.subplots()
        ax.plot(np.arange(epochs), loss)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Loss")
    
    def test(self, test_loader, verbose = True):
        self.model.training = False
        tn, fp, fn, tp, tpr, fpr, auc = self.test_step(test_loader)
        sens = tp/(tp + fn)
        spec = tn/(tn + fp)
        if(verbose):
            fig, ax = plt.subplots()
            ax.plot(fpr, tpr)
            ax.set_xlabel("False Positive Rate")
            ax.set_xlabel("True Positive Rate")
            ax.set_title("Confusion Matrix")
            print("Sensitivity: {:f}\tSpecificity: {:f}\tAUC: {:f}".format(sens, spec, auc))
        return sens, spec, auc