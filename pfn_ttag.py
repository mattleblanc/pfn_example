#  _____  ______ _   _ 
# |  __ \|  ____| \ | |
# | |__) | |__  |  \| |
# |  ___/|  __| | . ` |
# | |    | |    | |\  |
# |_|    |_|    |_| \_|
#  ________   __          __  __ _____  _      ______
# |  ____\ \ / /    /\   |  \/  |  __ \| |    |  ____|
# | |__   \ V /    /  \  | \  / | |__) | |    | |__
# |  __|   > <    / /\ \ | |\/| |  ___/| |    |  __|
# | |____ / . \  / ____ \| |  | | |    | |____| |____
# |______/_/ \_\/_/    \_\_|  |_|_|    |______|______|

# EnergyFlow - Python package for high-energy particle physics.
# Copyright (C) 2017-2020 Patrick T. Komiske III and Eric Metodiev

# standard library imports
from __future__ import absolute_import, division, print_function

# standard numerical library imports

# Data I/O and numerical imports
#import h5py
import numpy as np

# ML imports
import tensorflow as tf

from tensorflow.data import Dataset
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

# energyflow is not available by default
import energyflow as ef

# to fix keras version problem with energyflow
# via https://stackoverflow.com/questions/76650327/how-to-fix-cannot-import-name-version-from-tensorflow-keras
# from keras import __version__
# tf.keras.__version__ = __version__

from energyflow.archs.efn import PFN
from energyflow.datasets import qg_jets
from energyflow.utils import data_split, remap_pids, to_categorical

# Plotting imports
import matplotlib.pyplot as plt

print("pfn_example.py\tWelcome!")

################################### SETTINGS ###################################
# the commented values correspond to those in 1810.05165
###############################################################################

# data controls, can go up to 2000000 for full dataset
#train, val, test = 75000, 10000, 15000
train, val, test = 1500000, 250000, 250000
use_pids = False

# network architecture parameters
Phi_sizes, F_sizes = (100, 100, 128), (100, 100, 100)
# Phi_sizes, F_sizes = (100, 100, 256), (100, 100, 100)

# network training parameters
num_epoch = 500
batch_size = 500

from keras.callbacks import EarlyStopping,ModelCheckpoint
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=15)
mc = ModelCheckpoint('best_pfn.keras', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

################################################################################

print('Loading the dataset ...')

# load data
X, y = qg_jets.load(train + val + test)

print('Dataset loaded!')

# convert labels to categorical
Y = to_categorical(y, num_classes=2)

print('Loaded quark and gluon jets')

# preprocess by centering jets and normalizing pts
for x in X:
    mask = x[:,0] > 0
    yphi_avg = np.average(x[mask,1:3], weights=x[mask,0], axis=0)
    x[mask,1:3] -= yphi_avg
    x[mask,0] /= x[:,0].sum()

# handle particle id channel
if use_pids:
    remap_pids(X, pid_i=3)
else:
    X = X[:,:,:3]

print('Finished preprocessing')

# do train/val/test split 
(X_train, X_val, X_test,
 Y_train, Y_val, Y_test) = data_split(X, Y, val=val, test=test)

print('Done train/val/test split')
print('Model summary:')

# build architecture
pfn = PFN(input_dim=X.shape[-1], Phi_sizes=Phi_sizes, F_sizes=F_sizes)

# train model
pfn.fit(X_train, Y_train,
        epochs=num_epoch,
        batch_size=batch_size,
        validation_data=(X_val, Y_val),
        verbose=1,
        callbacks=[es,mc])

# get predictions on test data
preds = pfn.predict(X_test, batch_size=1000)

# get ROC curve
pfn_fp, pfn_tp, threshs = roc_curve(Y_test[:,1], preds[:,1])

# get area under the ROC curve
auc = roc_auc_score(Y_test[:,1], preds[:,1])
print()
print('PFN AUC:', auc)
print()

# get multiplicity and mass for comparison
masses = np.asarray([ef.ms_from_p4s(ef.p4s_from_ptyphims(x).sum(axis=0)) for x in X])
mults = np.asarray([np.count_nonzero(x[:,0]) for x in X])
mass_fp, mass_tp, threshs = roc_curve(Y[:,1], -masses)
mult_fp, mult_tp, threshs = roc_curve(Y[:,1], -mults)

# some nicer plot settings 
plt.rcParams['figure.figsize'] = (4,4)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.autolayout'] = True

# plot the ROC curves
plt.plot(pfn_tp, 1-pfn_fp, '-', color='black', label='PFN')
plt.plot(mass_tp, 1-mass_fp, '-', color='blue', label='Jet Mass')
plt.plot(mult_tp, 1-mult_fp, '-', color='red', label='Multiplicity')

# axes labels
plt.xlabel('Quark Jet Efficiency')
plt.ylabel('Gluon Jet Rejection')

# axes limits
plt.xlim(0, 1)
plt.ylim(0, 1)

# make legend and show plot
plt.legend(loc='lower left', frameon=False)
#plt.show()
plt.savefig('example_roc.pdf')
