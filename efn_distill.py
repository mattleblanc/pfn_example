"""                                   ____________________________________
         ____                        |.==================================,|
  ___   / __/____                    ||   I WILL LISTEN TO THE TEACHER.  ||
 / _ \ / /_ / __ \                   ||   I WILL LISTEN TO THE TEACHER.  ||
/  __// __// / / /                   ||   I WILL LISTEN TO THE TEACHER.  ||
\___//_/_ /_/ /_/  __   _  __ __     ||   I .----;ISTEN ,                ||
  ____/ /(_)_____ / /_ (_)/ // /     || /`\/      \/`\ /                 ||
 / __  // // ___// __// // // /      || )_/|      |\_(/\                 ||
/ /_/ // /(__  )/ /_ / // // /       ||    \______/  /\/                 ||
\__,_//_//____/ \__//_//_//_/        ||    _(____)__/ /                  ||
                                     ||___/ ,_ _  ___/___________________||
                                     '====\___\_) |======================='
                                          |______|
                                          /______\
                                           |_||_|
                                    jgs   (__)(__)
"""

# This script written by Matt LeBlanc (Brown University, 2024),
# adapted from pfn_example.py written by Patrick T. Komiske III and Eric Metodiev

# standard library imports
from __future__ import absolute_import, division, print_function
import argparse

# Data I/O and numerical imports
#import h5py
import numpy as np

# ML imports
import tensorflow as tf
import keras

tf.experimental.numpy.experimental_enable_numpy_behavior()

from tensorflow.data import Dataset
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

# energyflow is not available by default
import energyflow as ef

from energyflow.archs.efn import PFN
from energyflow.archs.efn import EFN
#from energyflow.archs.dnn import DNN
from energyflow.datasets import qg_jets
from energyflow.utils import data_split, remap_pids, to_categorical

# Plotting imports
import matplotlib.pyplot as plt

###########

# Function borrowed from Rikab
# https://github.com/rikab/GaussianAnsatz
from keras.layers import Dense, Dropout, Input, Concatenate
from keras.models import Model
def efn_input_converter(model_efn, shape=None, num_global_features=0):
    if num_global_features == 0:
        input_layer = Input(shape=shape)
        output = model_efn([input_layer[:, :, 0], input_layer[:, :, 1:]])
        return Model(input_layer, output)
    else:
        input_layer_1 = Input(shape=shape)
        input_layer_2 = Input(shape=(num_global_features,))
        output = model_efn([input_layer_1[:, :, 0], input_layer_1[:, :, 1:], input_layer_2])
        return Model([input_layer_1, input_layer_2], output)

# Distillation based on keras example by Kenneth Borup
# https://keras.io/examples/vision/knowledge_distillation/
# Adapted to PFN

class Distiller(tf.keras.Model):
    def __init__(self, student, teacher):
        super().__init__()
        self.teacher = teacher
        self.student = student
        self.loss_tracker = keras.metrics.Mean(name="distillation_loss")
        
    @property
    def metrics(self):
        metrics = super().metrics
        metrics.append(self.loss_tracker)
        return metrics
    
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
        # Forward pass of teacher
        teacher_predictions = self.teacher.model(x, training=False)
        student_loss = self.student_loss_fn(y, y_pred)
        
        with tf.GradientTape() as tape:
            # Forward pass of student
            #student_predictions = self.student.model([z,p], training=True)
            student_predictions = self.student(x, training=True)

            # Compute loss
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1),
            )

        """
        # MLB Note: This kind of combined-loss seems to break things at the moment...
        # just using distillation loss for now, but it works well!
            
        #student_loss  = self.student_loss_fn(y, y_pred) # Also compute the loss of training the student directly
        
        # alpha determines how much the student listens to the teacher or trusts itself
        #combined_loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
        """        

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        #gradients = tape.gradient(combined_loss, trainable_vars)
        gradients = tape.gradient(distillation_loss, trainable_vars)
        
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Report progress
        #self.loss_tracker.update_state(combined_loss)
        self.loss_tracker.update_state(distillation_loss)
        return {"distillation_loss": self.loss_tracker.result()} 
        #return loss

    def call(self, x):
        return self.student(x)

    
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
#train, val, test = 75000, 10000, 15000 # small
#train, val, test = 150000, 20000, 30000 # medium (2x small, ~0.1x complete)
#train, val, test = 1500000, 250000, 250000 # complete
frac_train, frac_val, frac_test = 0.75, 0.125, 0.125
#train, val, test = int(frac_train*1500000), int(frac_val*1500000), int(frac_test*1500000) # Only 1500000 jets in the H7 sample
train, val, test = int(frac_train*150000), int(frac_val*150000), int(frac_test*100000) # Using 10%
use_pids = args.usePIDs

# network architecture parameters
Phi_sizes, F_sizes = (100, 100, args.latentSize), (100, 100, 100)

# network training parameters
num_epoch = args.nEpochs
if(args.doEarlyStopping):
    num_epoch = 500
batch_size = args.batchSize

################################################################################

print('Loading the dataset ...')

# load data
X, y = qg_jets.load(train + val + test, generator='pythia', pad=True)
#X, y = qg_jets.load(train + val + test, generator='herwig', pad=True)

print('Dataset loaded!')
print(X.shape)

"""
The padded Pythia and Herwig datasets are different shapes:
   Pythia: (2000000, 148, 4)
   Herwig: (1500000, 154, 4)
 So, if you want to do comparative studies, you need to 
   e.g. pad the Pythia one by six entries along the second axis.
"""
#X = np.lib.pad(X, ((0,0), (0,6), (0,0)), mode='constant', constant_values=0)

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

#print(X.shape)
#print(X[0])

# do train/val/test split 
(X_train, X_val, X_test,
 Y_train, Y_val, Y_test) = data_split(X, Y, val=val, test=test)

# For PFN
#(X_train_PFN, X_val_PFN, X_test_PFN, Y_train_PFN, Y_val_PFN, Y_test_PFN) = data_split(XPFN, Y, val=val, test=test)

# For EFN
z_train = X_train[:,:,0]
z_val   = X_val[:,:,0]
z_test  = X_test[:,:,0]

p_train = X_train[:,:,1:]
p_val   = X_val[:,:,1:]
p_test  = X_test[:,:,1:]

# For EFN
#(z_train_EFN, z_val_EFN, z_test_EFN, 
# p_train_EFN, p_val_EFN, p_test_EFN,
# Y_train_EFN, Y_val_EFN, Y_test_EFN) = data_split(X[:,:,0], X[:,:,1:], Y, val=val, test=test)

print(p_train.shape, z_train.shape, X_train.shape)

print('Done train/val/test split')
print('Model summary:')

# build architecture
pfn_teacher = PFN(input_dim=X.shape[-1], Phi_sizes=Phi_sizes, F_sizes=F_sizes)

# train the teacher model
if(args.doEarlyStopping):
    from keras.callbacks import EarlyStopping,ModelCheckpoint
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=args.patience)
    mc = ModelCheckpoint('best_pfn_'+args.label+'_teacher.keras', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

    print("Training teacher:")
    pfn_teacher.fit(X_train, Y_train,
                    epochs=num_epoch,
                    batch_size=batch_size,
                    validation_data=(X_val, Y_val),
                    verbose=1,
                    callbacks=[es,mc])

else:
    print("Training teacher:")
    pfn_teacher.fit(X_train, Y_train,
                    epochs=num_epoch,
                    batch_size=batch_size,
                    validation_data=(X_val, Y_val),
                    verbose=1)

############################################

# Set up the 'simple' and 'student' models, which will be energy-flow networks (EFNs)

# build architecture, input_dim=2 (y,phi) for EFN

_efn_student = EFN(input_dim=2, Phi_sizes=Phi_sizes, F_sizes=F_sizes).model
_efn_simple  = EFN(input_dim=2, Phi_sizes=Phi_sizes, F_sizes=F_sizes).model

max_particles=X.shape[1]
efn_student = efn_input_converter(_efn_student,shape=(max_particles, 3))
efn_student.compile(loss="binary_crossentropy",
                    optimizer=tf.keras.optimizers.Adam())
efn_simple = efn_input_converter(_efn_simple,shape=(max_particles, 3))
efn_simple.compile(loss="binary_crossentropy",
                    optimizer=tf.keras.optimizers.Adam())

# train the simple model
if(args.doEarlyStopping):
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=args.patience)
    mc = ModelCheckpoint('best_dnn_'+args.label+'_simple.keras', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
    efn_simple.fit(X_train, Y_train,
                   epochs=num_epoch,
                   batch_size=batch_size,
                   validation_data=(X_val, Y_val),
                   verbose=1,
                   callbacks=[es,mc])
else:
    efn_simple.fit(X_train, Y_train,
                   epochs=num_epoch,
                   batch_size=batch_size,
                   validation_data=(X_val, Y_val),
                   verbose=1)
    
############################################

# train the student model

distiller = Distiller(student=efn_student, teacher=pfn_teacher)

distiller.compile(
    optimizer=tf.keras.optimizers.Adam(),
    metrics=[tf.keras.metrics.CategoricalCrossentropy()],
    student_loss_fn=tf.keras.losses.CategoricalCrossentropy(),
    distillation_loss_fn=tf.keras.losses.KLDivergence(),
    alpha=0.5, # was 0.1 but doesn't do anything right now
    temperature=3.0,
)

### MLB NOTE: Early stopping is not working yet for custom subclassed model ...
#   need to write custom callback function at some point, I think

# Distill teacher to student
#if(args.doEarlyStopping):
    #es_d = EarlyStopping(monitor='val_distillation_loss', mode='min', verbose=1, patience=args.patience)
    #mc_d = ModelCheckpoint('best_dnn_'+args.label+'_student.keras',
    #                       monitor='val_categorical_crossentropy',
    #                       mode='auto',
    #                       verbose=1,
    #                       save_best_only=True,
    #                       save_format="tf")
    
    #print("Training student:")
    #hist = distiller.fit(X_train,
    #              Y_train,#_for_distiller,
    #              epochs=num_epoch,
    #              batch_size=batch_size,
    #              validation_data=(X_val, Y_val),#_for_distiller),
    #              verbose=1,
    #              callbacks=[es_d])#,mc_d])
    
#else:
print("Training student:")
distiller.fit(X_train,
              Y_train,
              epochs=20, #num_epoch,
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
preds_student = efn_student.predict(X_test,
                                    batch_size=1000)
dnn_fp_student, dnn_tp_student, threshs_student = roc_curve(Y_test[:,1], preds_student[:,1])
auc_student  = roc_auc_score(Y_test[:,1], preds_student[:,1])
print()
print('Student PFN AUC:', auc_student)
print()

# get predictions on test data and ROC curve
preds_simple = efn_simple.predict(X_test,
                                  batch_size=1000)
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
