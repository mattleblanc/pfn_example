"""An example that makes an animation between two events using the EMD. Note
that `ffmpeg` must be installed in order for matplotlib to be able to render
the animation. Strange errors may result if there are issues with required
software components.

This version attempts to implement an even more generalized function for the animation, which would work for any array of jet events passed through.
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
keyframes3 = [events.particles[14930][:,:3], 
          events.particles[19751][:,:3],
          events.particles[12345][:,:3]]

keyframes7 = [events.particles[14930][:,:3], 
          events.particles[19751][:,:3],
          events.particles[12345][:,:3],
          events.particles[45465][:,:3],
          events.particles[19816][:,:3],
          events.particles[21984][:,:3],
          events.particles[12123][:,:3]]

# center the jets
# event0[:,1:3] -= np.average(event0[:,1:3], weights=event0[:,0], axis=0)
# event1[:,1:3] -= np.average(event1[:,1:3], weights=event1[:,0], axis=0)
# event2[:,1:3] -= np.average(event2[:,1:3], weights=event2[:,0], axis=0)

## prepare the jets for any amount of events
kfs = []
def prepare_events(keyframes):
    ## center the jets by y, phi (elements 1-2 in particles list)
    for event in keyframes:
        event[:,1:3] -= np.average(event[:,1:3], weights=event[:,0], axis=0) 

    ## mask out particles outside of the cone (radius R)
    for event in keyframes:
        event = event[np.linalg.norm(event[:,1:3], axis=1) < R]

    ## copy events list to a numpy 
    global kfs
    for ev in keyframes:
        kfs.append(np.copy(ev))

# remove this next line once I figure out what's going on
prepare_events(keyframes7)

print(kfs)

#############################################################
# MAKE ANIMATION
#############################################################

fig, ax = plt.subplots()
    
merged = merge(kfs[0], kfs[1], lamb=0, R=R)

## assign initial pts, ys, phis based on first keyframe
pts0, ys0, phis0 = merged[:,0], merged[:,1], merged[:,2]

## initialize scatterplot with first keyframe
scatter = ax.scatter(ys0, phis0, color='blue', s=pts0, lw=0)

## define the current phase which the smart_animate function uses
current_phase = 0

## smart animate function, which is called sequentially
def smart_animate(i):
    # still might want to add another parameter?

    # clear ax before each frame drawing
    ax.clear()

    # call prepare_events on the keyframes list
    # prepare_events(keyframes)

    # need 2 times the number of keyframes for transition stages
    nstages = 2 * len(kfs)

    # stage number based on frames
    stage_size = (nframes / nstages)

    # current keyframe number
    global current_phase
    current_kf = int(np.floor(current_phase/2))

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
        current_phase += 1
    
    color = 'blue' # change this later
    

    merged = merge(ev0, ev1, lamb=lamb, R=0.5)
    pts, ys, phis = merged[:,0], merged[:,1], merged[:,2]
    scatter = ax.scatter(ys, phis, color=color, s=zf*pts, lw=0)

    ax.set_xlim(-R, R); ax.set_ylim(-R, R);
    ax.set_axis_off()

    return scatter,

anim = animation.FuncAnimation(fig, smart_animate, frames=nframes, repeat=True)
anim.save('smartanimationgen.gif', fps=fps, dpi=200)

# uncomment these lines if running in a jupyter notebook
# from IPython.display import HTML
# HTML(anim.to_html5_video())
