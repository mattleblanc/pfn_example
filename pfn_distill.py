"""         ____                
    ____   / __/____            
   / __ \ / /_ / __ \           
  / /_/ // __// / / /           
 / .___//_/  /_/ /_/            
/_/    __ _        __   _  __ __
  ____/ /(_)_____ / /_ (_)/ // /
 / __  // // ___// __// // // / 
/ /_/ // /(__  )/ /_ / // // /  
\__,_//_//____/ \__//_//_//_/                                   
"""

# This script written by Matt LeBlanc (Brown University, 2024),
# adapted from pfn_example.py written by Patrick T. Komiske III and Eric Metodiev

# EnergyFlow - Python package for high-energy particle physics.
# Copyright (C) 2017-2020 Patrick T. Komiske III and Eric Metodiev

# standard library imports
from __future__ import absolute_import, division, print_function
import argparse

# Data I/O and numerical imports
#import h5py
import numpy as np

# ML imports
import tensorflow as tf
import keras

from tensorflow.data import Dataset
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

# energyflow is not available by default
import energyflow as ef

from energyflow.archs.efn import PFN
from energyflow.archs.dnn import DNN
from energyflow.datasets import qg_jets
from energyflow.utils import data_split, remap_pids, to_categorical

# Plotting imports
import matplotlib.pyplot as plt

###########

# Distillation based on keras example by Kenneth Borup
# https://keras.io/examples/vision/knowledge_distillation/
# Adapted to PFN

class Distiller(tf.keras.Model):
    def __init__(self, student, teacher):
        super().__init__()
        self.teacher = teacher
        self.student = student

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.1,
        temperature=3,
    ):
        """Configure the distiller.

        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def compute_loss(
        self, x=None, y=None, y_pred=None, sample_weight=None, allow_empty=False
    ):
        
        teacher_pred = self.teacher.model(x, training=False)
        student_loss = self.student_loss_fn(y, y_pred)
        
        distillation_loss = self.distillation_loss_fn(
            tf.nn.softmax(teacher_pred / self.temperature, axis=1),
            tf.nn.softmax(y_pred / self.temperature, axis=1),            
        ) * (self.temperature**2)
        #distillation_loss = self.distillation_loss_fn( teacher_pred, y_pred )

        #loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
        loss = distillation_loss

        return loss

    def call(self, x):
        tf.experimental.numpy.experimental_enable_numpy_behavior()
        return self.student.model(x.reshape(-1,X_train.shape[1]*X_train.shape[2]))
        #return self.student(x)
        #return self.student.model(x) # needed for EFN/PFN

###########
# Main script

parser = argparse.ArgumentParser(description="%prog [options]", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--nEpochs", dest='nEpochs', default=0, type=int, required=False,
                    help="How many epochs to train for?")
parser.add_argument("--batchSize", dest='batchSize', default=500, type=int, required=False,
                    help="How large should the batch size be?")
parser.add_argument("--latentSize", dest='latentSize', default=128, type=int, required=False,
                    help="What is the dimension of the per-particle embedding? n.b. must be a power of 2!")
parser.add_argument("--doEarlyStopping", dest='doEarlyStopping', action='store_true', required=False,
                    help="Do early stopping?")
parser.add_argument("--patience", dest='patience', default=10, type=int, required=False,
                    help="How patient?")
parser.add_argument("--usePIDs", dest='usePIDs', action='store_true', required=False,
                    help="Use PIDs? If True, this script will currently break!")
parser.add_argument("--makeROCs", dest='makeROCs', action='store_true', required=False,
                    help="Make ROC curves?")
parser.add_argument("--label", dest='label', type=str, required=False,
                    help="Label for output")
args = parser.parse_args()

if(args.nEpochs==0 and args.doEarlyStopping==False):
    raise Exception("You need to specify a number of epochs to train for, or to use early stopping!")

# nice : https://stackoverflow.com/questions/57025836/how-to-check-if-a-given-number-is-a-power-of-two
if( not(args.latentSize & (args.latentSize-1))==0 and args.latentSize!=0):
    raise Exception("The dimension of the per-particle embedding has to be a power of 2!")

if(args.nEpochs>0 and args.doEarlyStopping):
    raise Exception("Either specify early stopping, **or** a number of epochs to train for!")

print("pfn_example.py\tWelcome!")

################################### SETTINGS ###################################
# the commented values correspond to those in 1810.05165
###############################################################################

# data controls, can go up to 2000000 for full dataset
#train, val, test = 75000, 10000, 15000
#train, val, test = 150000, 20000, 30000
train, val, test = 1500000, 250000, 250000
use_pids = args.usePIDs

# network architecture parameters
Phi_sizes_teacher, F_sizes_teacher = (100, 100, args.latentSize), (100, 100, 100)
Phi_sizes_student, F_sizes_student = (100, 100, args.latentSize/8), (100, 100, 100)
Phi_sizes_simple, F_sizes_simple   = (100, 100, args.latentSize/8), (100, 100, 100)
# Phi_sizes, F_sizes = (100, 100, 256), (100, 100, 100)

# network training parameters
num_epoch = args.nEpochs
if(args.doEarlyStopping):
    num_epoch = 500
batch_size = args.batchSize

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
pfn_teacher = PFN(input_dim=X.shape[-1], Phi_sizes=Phi_sizes_teacher, F_sizes=F_sizes_teacher)
#pfn_student = PFN(input_dim=X.shape[-1], Phi_sizes=Phi_sizes_student, F_sizes=F_sizes_student)
#pfn_simple = PFN(input_dim=X.shape[-1], Phi_sizes=Phi_sizes_simple, F_sizes=F_sizes_simple)

# train the teacher model

if(args.doEarlyStopping):
    from keras.callbacks import EarlyStopping,ModelCheckpoint
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=args.patience)
    mc = ModelCheckpoint('best_pfn_'+args.label+'.keras', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

    print("Training teacher:")
    pfn_teacher.fit(X_train, Y_train,
                    epochs=num_epoch,
                    batch_size=batch_size,
                    validation_data=(X_val, Y_val),
                    verbose=1,
                    callbacks=[es,mc])
    
    #print("Training simple:")
    #pfn_simple.fit(X_train, Y_train,
    #               epochs=num_epoch,
    #               batch_size=batch_size,
    #               validation_data=(X_val, Y_val),
    #               verbose=1,
    #               callbacks=[es,mc])

else:
    print("Training teacher:")
    pfn_teacher.fit(X_train, Y_train,
                    epochs=num_epoch,
                    batch_size=batch_size,
                    validation_data=(X_val, Y_val),
                    verbose=1)

    #print("Training simple:")
    #pfn_simple.fit(X_train, Y_train,
    #               epochs=num_epoch,
    #               batch_size=batch_size,
    #               validation_data=(X_val, Y_val),
    #               verbose=1)

############################################

# Set up the 'simple' and 'student' models, which will be simple fully-connected DNNs

# build architecture

print(X_train.shape)
print(X_train.shape[1:])
print(X_train.shape[1:2])

dense_sizes = (100, 100)
dnn_simple  = DNN(input_dim=X_train.shape[1]*X_train.shape[2], dense_sizes=dense_sizes)
dnn_student = DNN(input_dim=X_train.shape[1]*X_train.shape[2], dense_sizes=dense_sizes)

# train model
dnn_simple.fit(X_train.reshape(-1,X_train.shape[1]*X_train.shape[2]), Y_train,
               epochs=num_epoch,
               batch_size=batch_size,
               validation_data=(X_val.reshape(-1,X_val.shape[1]*X_val.shape[2]), Y_val),
               verbose=1)

############################################

# Train the 'simple' model for comparison later

"""
if(args.doEarlyStopping):
    from keras.callbacks import EarlyStopping,ModelCheckpoint
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=args.patience)
    mc = ModelCheckpoint('best_pfn_'+args.label+'.keras', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
    
    print("Training simple:")
    dnn_simple.fit(X_train, Y_train_for_distiller,
                   epochs=num_epoch,
                   batch_size=batch_size,
                   validation_data=(X_val, Y_val_for_distiller),
                   verbose=1,
                   callbacks=[es,mc])

else:
    print("Training simple:")
    dnn_simple.fit(X_train, Y_train_for_distiller,
                   epochs=num_epoch,
                   batch_size=batch_size,
                   validation_data=(X_val, Y_val_for_distiller),
                   verbose=1)
"""

############################################
    
# train the student model

distiller = Distiller(student=dnn_student, teacher=pfn_teacher)

distiller.compile(
    optimizer=tf.keras.optimizers.Adam(),
    #metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    #student_loss_fn=tf.keras.losses.SparseCategoricalCrossentropy(),#from_logits=True),
    metrics=[tf.keras.metrics.CategoricalCrossentropy()],
    student_loss_fn=tf.keras.losses.CategoricalCrossentropy(),
    distillation_loss_fn=tf.keras.losses.KLDivergence(),
    alpha=1.0, # was 0.1
    temperature=3.0,
)

# Distiller loss functions do *not* want one-hot encoding ...
#Y_train_for_distiller = np.asarray([row[0] for row in Y_train])
#Y_val_for_distiller = np.asarray([row[0] for row in Y_val])

# Distill teacher to student
if(args.doEarlyStopping):
    #from keras.callbacks import EarlyStopping,ModelCheckpoint
    es_d = EarlyStopping(monitor='val_categorical_crossentropy', mode='min', verbose=1, patience=args.patience)
    mc_d = ModelCheckpoint('best_pfn_'+args.label+'.keras', monitor='val_categorical_crossentropy', mode='max', verbose=1, save_best_only=True)

    print("Training student:")
    distiller.fit(X_train,
                  Y_train,#_for_distiller,
                  epochs=num_epoch,
                  batch_size=batch_size,
                  validation_data=(X_val, Y_val),#_for_distiller),
                  verbose=1,
                  callbacks=[es_d,mc_d])
else:
    print("Training student:")
    distiller.fit(X_train,
                  Y_train,#,_for_distiller,
                  epochs=num_epoch,
                  batch_size=batch_size,
                  validation_data=(X_val, Y_val),#_for_distiller),
                  verbose=1)
#########################################################################

    
# get predictions on test data and ROC curve
preds_teacher = pfn_teacher.predict(X_test, batch_size=1000)
pfn_fp_teacher, pfn_tp_teacher, threshs_teacher = roc_curve(Y_test[:,1], preds_teacher[:,1])
auc_teacher  = roc_auc_score(Y_test[:,1], preds_teacher[:,1])
print()
print('Teacher PFN AUC:', auc_teacher)
print()

# get predictions on test data and ROC curve
preds_student = dnn_student.predict(X_test.reshape(-1,X_val.shape[1]*X_val.shape[2]), batch_size=1000)
dnn_fp_student, dnn_tp_student, threshs_student = roc_curve(Y_test[:,1], preds_student[:,1])
auc_student  = roc_auc_score(Y_test[:,1], preds_student[:,1])
print()
print('Student PFN AUC:', auc_student)
print()

# get predictions on test data and ROC curve
preds_simple = dnn_simple.predict(X_test.reshape(-1,X_val.shape[1]*X_val.shape[2]), batch_size=1000)
dnn_fp_simple, dnn_tp_simple, threshs_simple = roc_curve(Y_test[:,1], preds_simple[:,1])
auc_simple  = roc_auc_score(Y_test[:,1], preds_simple[:,1])
print()
print('Simple PFN AUC:', auc_simple)
print()

if(args.makeROCs):

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
    plt.plot(pfn_tp_teacher, 1-pfn_fp_teacher, '-', color='black', label='Teacher PFN (AUC: '+str(round(auc_teacher,3))+')')
    plt.plot(dnn_tp_student, 1-dnn_fp_student, '-', color='black', label='Student PFN (AUC: '+str(round(auc_student,3))+')')
    plt.plot(dnn_tp_simple,  1-dnn_fp_simple, '-', color='black', label='Simple PFN (AUC: '+str(round(auc_simple,3))+')')        
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
    plt.savefig('roc_'+args.label+'.pdf')
