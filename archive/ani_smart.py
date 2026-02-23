"""An example that makes an animation between two events using the EMD. Note
that `ffmpeg` must be installed in order for matplotlib to be able to render
the animation. Strange errors may result if there are issues with required
software components.

This version attempts to implement a generalized function for the animation.
"""

#           _   _ _____ __  __       _______ _____ ____  _   _
#     /\   | \ | |_   _|  \/  |   /\|__   __|_   _/ __ \| \ | |
#    /  \  |  \| | | | | \  / |  /  \  | |    | || |  | |  \| |
#   / /\ \ | . ` | | | | |\/| | / /\ \ | |    | || |  | | . ` |
#  / ____ \| |\  |_| |_| |  | |/ ____ \| |   _| || |__| | |\  |
# /_/    \_\_| \_|_____|_|  |_/_/    \_\_|  |_____\____/|_| \_|
#  ________   __          __  __ _____  _      ______
# |  ____\ \ / /    /\   |  \/  |  __ \| |    |  ____|
# | |__   \ V /    /  \  | \  / | |__) | |    | |__
# |  __|   > <    / /\ \ | |\/| |  ___/| |    |  __|
# | |____ / . \  / ____ \| |  | | |    | |____| |____
# |______/_/ \_\/_/    \_\_|  |_|_|    |______|______|

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

import energyflow as ef
from matplotlib import animation, rc

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
zf = 2           # size of points in scatter plot
lw = 1           # linewidth of flow lines
fps = 40         # frames per second, increase this for sharper resolution
nframes = 10*fps # total number of frames - originally 10*fps, but might need to change
R = 0.5          # jet radius


#############################################################
# LOAD IN JETS
#############################################################
specs = ['375 <= corr_jet_pts <= 425', 'abs_jet_eta < 1.9', 'quality >= 2']
events = ef.mod.load(*specs, dataset='cms', amount=0.01)

## list of events to be displayed initialized here, in order of display in the animation
# particle [pT,y,phi]
keyframes = [events.particles[14930][:,:3], 
          events.particles[19751][:,:3],
          events.particles[12345][:,:3]]

# center the jets
# event0[:,1:3] -= np.average(event0[:,1:3], weights=event0[:,0], axis=0)
# event1[:,1:3] -= np.average(event1[:,1:3], weights=event1[:,0], axis=0)
# event2[:,1:3] -= np.average(event2[:,1:3], weights=event2[:,0], axis=0)

## center the jets by y, phi (elements 1-2 in particles list)
for event in keyframes:
    event[:,1:3] -= np.average(event[:,1:3], weights=event[:,0], axis=0) 

# mask out particles outside of the cone
# event0 = event0[np.linalg.norm(event0[:,1:3], axis=1) < R]
# event1 = event1[np.linalg.norm(event1[:,1:3], axis=1) < R]
# event2 = event2[np.linalg.norm(event2[:,1:3], axis=1) < R]

## mask out particles outside of the cone (radius R)
for event in keyframes:
    event = event[np.linalg.norm(event[:,1:3], axis=1) < R]

# print(keyframes)
## copy events list to a numpy 
kfs = []
for ev in keyframes:
    kfs.append(np.copy(ev))


#############################################################
# MAKE ANIMATION
#############################################################

fig, ax = plt.subplots()

# merged0 = merge(ev0, ev1, lamb=0, R=R)
# merged1 = merge(ev1, ev2, lamb=0, R=R)
# merged2 = merge(ev2, ev0, lamb=0, R=R)

## find merge arrays for all event pairs, including one for a clean loop (last->first event merge)
# merges = list(len(kfs))
# merges = []
# for i in range(0, len(kfs) - 1):
#     if i < (len(kfs) - 1):
#         ev0 = kfs[i]
#         ev1 = kfs[i+1]
#         merges.append(merge(ev0, ev1, lamb=0, R=R))
#     elif i == (len(kfs) - 1):
#         ev0 = kfs[i]
#         ev1 = kfs[0]
#         merges.append(merge(ev0, ev1, lamb=0, R=R))
#     else:
#         raise Exception("You don't need to animate something with less than 2 frames...")
    
merged = merge(kfs[0], kfs[1], lamb=0, R=R)
## sanity check - delete later
# print(kfs)

## assign initial pts, ys, phis based on first keyframe
# pts0, ys0, phis0 = merges[0][:,0], merges[0][:,1], merges[0][:,2]
pts0, ys0, phis0 = merged[:,0], merged[:,1], merged[:,2]

# pts0, ys0, phis0 = merged0[:,0], merged0[:,1], merged0[:,2]
# pts1, ys1, phis1 = merged1[:,0], merged1[:,1], merged1[:,2]
# pts2, ys2, phis2 = merged2[:,0], merged2[:,1], merged2[:,2]

## initialize scatterplot with first keyframe
scatter = ax.scatter(ys0, phis0, color='blue', s=pts0, lw=0)

## define the current phase which the smart_animate function uses
current_phase = 0

## smart animate function, which is called sequentially
def smart_animate(i):
    # events list is currently hardcoded in, but might need to add a param to smart_animate that takes the list
    # also need to make the animation go like /\ as original did instead of like D (12321 instead of 1231)
    
    # clear ax before each frame drawing
    ax.clear()

    # need 2 times the number of keyframes for transition stages
    nstages = 2 * len(kfs)

    # stage number based on frames
    stage_size = (nframes / nstages)

    # current keyframe number
    global current_phase
    current_kf = int(np.floor(current_phase/2))
    # print(stage_size)

    # assuming i starts indexing at 0,
    lamb = (nstages*(i - (current_phase * stage_size))) / (nframes-1)

    # even phases are the static images of keyframes
    if (current_phase % 2) == 0:
        print('even phase')
        ev0 = kfs[current_kf]
        ev1 = kfs[current_kf]

    # odd phases are transitions between keyframes
    elif (current_phase % 2) == 1:
        print('odd phase')
        if current_phase == (nstages - 1):
            ev0 = kfs[0]
            ev1 = kfs[current_kf]
        else:
            ev0 = kfs[current_kf + 1]
            ev1 = kfs[current_kf]

    print('phase',current_phase)
    print('keyframe',current_kf)
    print('frame',i)

    # set modulo to recognize when the phase ends
    if ((i+1) % stage_size) < 1: # not == due to non-integer stage_size
        # if i == (nframes - 1):
        #     print('last frame!')
        #     current_phase = 0
        # else:
        current_phase += 1
    
    color = 'blue' # change this later
    

    merged = merge(ev0, ev1, lamb=lamb, R=0.5)
    pts, ys, phis = merged[:,0], merged[:,1], merged[:,2]
    scatter = ax.scatter(ys, phis, color=color, s=zf*pts, lw=0)

    ax.set_xlim(-R, R); ax.set_ylim(-R, R);
    ax.set_axis_off()

    return scatter,

anim = animation.FuncAnimation(fig, smart_animate, frames=nframes, repeat=True)
anim.save('smartanimation.gif', fps=fps, dpi=200)

# uncomment these lines if running in a jupyter notebook
# from IPython.display import HTML
# HTML(anim.to_html5_video())
