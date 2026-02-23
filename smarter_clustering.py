"""An example that makes an animation between two events using the EMD. Note
that `ffmpeg` must be installed in order for matplotlib to be able to render
the animation. Strange errors may result if there are issues with required
software components.

This version attempts to convert the general animation function (for any array of jet events passed through) from 'gen_ani_smart.py', but replacing the jet events with each step in any of the 3 jet declustering algorithms (KT, AKT, CA), which are plotted in 'fastjet-OnlyHistory-##.py'. Based on 'smart_clustering.py' but with actual options and cleaned up!
"""

# EnergyFlow - Python package for high-energy particle physics.
# Copyright (C) 2017-2022 Patrick T. Komiske III and Eric Metodiev

# standard library imports
from __future__ import absolute_import, division, print_function

# standard numerical library imports
import numpy as np

# matplotlib is required for this example
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (4,4)

#############################################################
# NOTE: ffmpeg must be installed
# on macOS this can be done with `brew install ffmpeg`
# on Ubuntu this would be `sudo apt-get install ffmpeg`
#############################################################

# on windows, the following might need to be uncommented
# plt.rcParams['animation.ffmpeg_path'] = 'C:\\ffmpeg\\bin\\ffmpeg.exe'

#############################################################
# PYTHIA SETUP
#############################################################
# pythia imports
import os
import pythia8
# produce leading-order events with pythia
pythia = pythia8.Pythia()
pythia.readString("Beams:eCM = 14000.") # energy of collisions
pythia.readString("HardQCD:all = on")
pythia.readString("PhaseSpace:pTHatMin = 500."); # filter on minimum event energy
# the following is a noisy line but useful to confirm pythia setup
pythia.init();

import energyflow as ef
from matplotlib import animation, rc
import fastjet as fj

# helper function to interpolate between the optimal transport of two events
def merge(ev0, ev1, R=1, lamb=0.5):
    emd, G = ef.emd.emd(ev0, ev1, R=R, return_flow=True)

    merged = []
    for i in range(len(ev0)):
        for j in range(len(ev1)):
            if G[i, j] > 0:
                merged.append([G[i,j], lamb*ev0[i,1] + (1-lamb)*ev1[j,1],
                                       lamb*ev0[i,2] + (1-lamb)*ev1[j,2]])

    # detect which event has more pT
    if np.sum(ev0[:,0]) > np.sum(ev1[:,0]):
        for i in range(len(ev0)):
            if G[i,-1] > 0:
                merged.append([G[i,-1]*lamb, ev0[i,1], ev0[i,2]])
    else:
        for j in range(len(ev1)):
            if G[-1,j] > 0:
                merged.append([G[-1,j]*(1-lamb), ev1[j,1], ev1[j,2]])

    return np.asarray(merged)


#############################################################
# ANIMATION OPTIONS
#############################################################
zf = 20           # size of points in scatter plot, originally 2
lw = 0           # linewidth of flow lines, originally 1
fps = 60         # frames per second, increase this for sharper resolution
nframes = 10*fps # total number of frames - originally 10*fps, but might need to change
R = 0.4          # jet radius, originally 0.5

#############################################################
# GENERATE LEADING-ORDER EVENTS
#############################################################
n_events = 1
particle_y_cut = 4.9 # cut on hadron rapidity
particle_pt_cut = 2.0 # cut on hadron pT, originally 0.5
min_jet_pt = 50.0 # cut on jet pT
## need to make R=0.4 jets with various algorithms: (here R is above)
kt_jetdef  = fj.JetDefinition(fj.kt_algorithm,        R)
akt_jetdef = fj.JetDefinition(fj.antikt_algorithm,    R)
ca_jetdef  = fj.JetDefinition(fj.cambridge_algorithm, R) # algorithm, R
## toggle through generations (???) - seven seems to give one where 3 algos produce marginally different leading jets
pythia.next()
pythia.next()
pythia.next()
pythia.next()
pythia.next()
pythia.next()
pythia.next()

#############################################################
# GENERATE LEADING-ORDER EVENTS
#############################################################
## in order to use particles and jets with wasserstein & sklearn, we'll use 0-padded np.arrays
max_n_jets = 2
max_n_particles = 200
## np.empty gives 0-pads ...
py_events_weights   = np.zeros((n_events, 1))
py_events_jets      = np.zeros((n_events, max_n_jets, 4))
py_events_particles = np.zeros((n_events, max_n_particles, 3))
## outer loop over events
for idx_event,_ in enumerate(range(n_events)):
    if not pythia.next():
        continue
    ## create particles list (format: PseudoJet<pt,eta,phi>??)
    particles = []
    pidx = 0
    for idx,p in enumerate(pythia.event):
        ## apply y and pT cuts
        if p.isFinal() and p.isHadron() and (abs(p.y()) < particle_y_cut) and (p.pT() > particle_pt_cut):
            pj = fj.PseudoJet(p.px(), p.py(), p.pz(), p.e())
            particles.append(pj)
            ## only add if haven't exceeded the length of event particles(?)
            if(pidx<max_n_particles):
                py_events_particles[idx_event, pidx, 0] = pj.pt()
                py_events_particles[idx_event, pidx, 1] = pj.eta()
                py_events_particles[idx_event, pidx, 2] = pj.phi()
            ## advance particles counter
            pidx += 1
    ## make the jets
    kt_cluster  = fj.ClusterSequence(particles, kt_jetdef)
    akt_cluster = fj.ClusterSequence(particles, akt_jetdef)
    ca_cluster  = fj.ClusterSequence(particles, ca_jetdef)

#############################################################
# GET THE JETS FROM EACH ALGO, SORTED BY pT
#############################################################
kt_jets  = fj.sorted_by_pt(kt_cluster.inclusive_jets(min_jet_pt))
akt_jets = fj.sorted_by_pt(akt_cluster.inclusive_jets(min_jet_pt))
ca_jets  = fj.sorted_by_pt(ca_cluster.inclusive_jets(min_jet_pt))
print("Clustered with "+kt_jetdef.description())
print("The leading jet pT is: "+str(kt_jets[0].perp())+" GeV\n")
print("Clustered with "+akt_jetdef.description())
print("The akt jet pT is: "+str(akt_jets[0].perp())+" GeV\n")
print("Clustered with "+ca_jetdef.description())
print("The CA jet pT is: "+str(ca_jets[0].perp())+" GeV\n")


#############################################################
# CONSTITUENTS FOR THE LEADING JET BY ALGORITHM
#############################################################
# function for preparing plot
def prep_plots(this_algo):
    # this_algo is one of "kt", "akt", "ca" (changed by user at bottom of this file)
    for i in range(0,1):
        ## get the constituents of the jet (kt_jets, akt_jets, ca_jets)
        if this_algo=="kt":
            constituents = fj.sorted_by_pt(kt_jets[i].constituents())
        elif this_algo=="akt":
            constituents = fj.sorted_by_pt(akt_jets[i].constituents())
        elif this_algo=="ca":
            constituents = fj.sorted_by_pt(ca_jets[i].constituents())
        for j in range(0,len(constituents)) :
            print(j,"\t",
                "%0.4f"%constituents[j].rap(),"\t",
                "%0.4f"%constituents[j].phi(),"\t",
                "%0.4f"%constituents[j].perp(),"\t",
                );
        ## prepare for later plotting
        global avg_rap
        global avg_phi
        avg_rap = sum([c.rap() for c in constituents]) / len(constituents)
        avg_phi = sum([c.phi() for c in constituents]) / len(constituents)


#############################################################
# LOAD PSEUDOJETS DECLUSTERING HISTORY BY ALGORITHM
#############################################################
## create list of fastjet pseudojets
kfs = []
def cluster_history(algo):
    """algo:string of either kt, akt, or ca; returns a list of pseudojet events, each event being a step in the declustering history"""
    hlst = []
    if algo == "kt":
        for j in range(0,len(kt_jets[0].constituents())):
            kt_xsubjs = kt_cluster.exclusive_subjets_up_to(kt_jets[0], j)
            this_steps_plst = []
            for xsj in kt_xsubjs:
                this_steps_plst.append([xsj.perp(), xsj.rap()-avg_rap, xsj.phi()-avg_phi])
            hlst.append(this_steps_plst)
    elif algo == "akt":
        for j in range(0,len(akt_jets[0].constituents())):
            akt_xsubjs = akt_cluster.exclusive_subjets_up_to(akt_jets[0], j)
            this_steps_plst = []
            for xsj in akt_xsubjs:
                this_steps_plst.append([xsj.perp(), xsj.rap()-avg_rap, xsj.phi()-avg_phi])
            hlst.append(this_steps_plst)
    elif algo == "ca":
        for j in range(0,len(ca_jets[0].constituents())):
            ca_xsubjs = ca_cluster.exclusive_subjets_up_to(ca_jets[0], j)
            this_steps_plst = []
            for xsj in ca_xsubjs:
                this_steps_plst.append([xsj.perp(), xsj.rap()-avg_rap, xsj.phi()-avg_phi])
            hlst.append(this_steps_plst)
    else:
        print("not a valid algorithm")

    ## need to pop the first step off (empty array)
    hlst.pop(0)
    ## reverse list to get from subjets to jet
    hlst.reverse()
    ## copy events list to a numpy
    kfs = []
    for step in hlst:
        kfs.append(np.copy(step))
    return kfs


#############################################################
# MAKE ANIMATION
#############################################################
## animation settings
fig, ax = plt.subplots()
alpha = 0.3
color = 'black' # default, it shouldn't show

# set the algorithm and generate clustering history
this_algo = "ca" # ("kt", "akt", or "ca") <- CHANGE THIS STRING!

prep_plots(this_algo)
# colors to differentiate between algorithms at first glance
kfs = cluster_history(this_algo)
if this_algo=="kt":
    color = 'red'
elif this_algo=="akt":
    color = 'green'
elif this_algo=="ca":
    color = 'blue'
    
merged = merge(kfs[0], kfs[1], lamb=0, R=R)

## assign initial pts, ys, phis based on first keyframe
pts0, ys0, phis0 = merged[:,0], merged[:,1], merged[:,2]

## initialize scatterplot with first keyframe
scatter = ax.scatter(ys0, phis0, color=color, s=pts0, lw=lw)

## define the current phase which the smart_animate function uses
current_phase = 0

## smart animate function, which is called sequentially
def smart_animate(i):
    ## clear ax before each frame drawing
    ax.clear()
    ## need 2 times the number of keyframes for transition stages
    nstages = 2 * len(kfs)
    ## stage number based on frames
    stage_size = (nframes / nstages)
    ## current keyframe number
    global current_phase
    current_kf = int(np.floor(current_phase/2))
    ## assuming i starts indexing at 0,
    lamb = (nstages*(i - (current_phase * stage_size))) / (nframes-1)

    ## even phases are the static images of keyframes
    if (current_phase % 2) == 0:
        print('even phase')
        ev0 = kfs[current_kf]
        ev1 = kfs[current_kf]
        # alpha = 0.2
    ## odd phases are transitions between keyframes
    elif (current_phase % 2) == 1:
        print('odd phase')
        if current_phase == (nstages - 1):
            ev0 = kfs[0]
            ev1 = kfs[current_kf]
        else:
            ev0 = kfs[current_kf + 1]
            ev1 = kfs[current_kf]
        # alpha = 1
    ## confirm phases and frames
    print('phase',current_phase)
    print('keyframe',current_kf)
    print('frame',i)
    ## set modulo to recognize when the phase ends
    if ((i+1) % stage_size) < 1: # not == due to non-integer stage_size
        current_phase += 1

    ## interpolate and scatter!
    merged = merge(ev0, ev1, lamb=lamb, R=0.5)
    pts, ys, phis = merged[:,0], merged[:,1], merged[:,2]
    scatter = ax.scatter(ys, phis, color=color, s=zf*np.log(pts), alpha=alpha, lw=lw)

    ## fix limits somehow ???
    # ax.set_xlim(-R, R); ax.set_ylim(-R, R);
    ax.set_xlim(-R-.1, R+.1); ax.set_ylim(-R-.1, R+.1);
    # ax.set_axis_off()

    return scatter,

# nframes = len(kt_jets[0].constituents()) - 1

anim = animation.FuncAnimation(fig, smart_animate, frames=nframes, repeat=True)
anim.save('smarterclustering_'+this_algo+'.gif', fps=fps, dpi=200)
print("Completed cluster animation with " + this_algo + " algorithm.")

# uncomment these lines if running in a jupyter notebook
# from IPython.display import HTML
# HTML(anim.to_html5_video())