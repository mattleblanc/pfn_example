### PARTS COPIED FROM 'fastjet.ipynb'
## Here, code involving figures/plots is updated in order to hopefully save multiple figures using a single python file (converting from the notebook structure of the original code). This might help with doing the final animation.

import os
import matplotlib.pyplot as plt
import numpy as np

import pythia8

# Produce leading-order events with Pythia.
pythia = pythia8.Pythia()
pythia.readString("Beams:eCM = 14000.") # Energy of collisions

pythia.readString("HardQCD:all = on")
pythia.readString("PhaseSpace:pTHatMin = 500."); # Filter on minimum event energy

pythia.init(); # this is a noisy line but useful to confirm how Pythia is set up

import fastjet as fj

#############################################################
# DEFINE ALGORITHMS
#############################################################
n_events = 1
particle_y_cut = 4.9 # cut on hadron rapidity
particle_pt_cut = 0.5 # cut on hadron pT
min_jet_pt = 50.0 #cut on jet pT

# We will want to make R=0.4 jets with various algorithms
kt_jetdef  = fj.JetDefinition(fj.kt_algorithm,        0.4) 
akt_jetdef = fj.JetDefinition(fj.antikt_algorithm,    0.4)
ca_jetdef  = fj.JetDefinition(fj.cambridge_algorithm, 0.4) # algorithm, R

# when running pythia in a notebook - do we need here?
# pythia.next()

#############################################################
# GENERATE LEADING-ORDER EVENTS
#############################################################
# We'll store the particles and jets for further study.
# In order to use them with wasserstein & sklearn, we'll use 0-padded np.arrays
max_n_jets = 2
max_n_particles = 200
# np.empty gives 0-pads ...
py_events_weights = np.zeros((n_events,1))
py_events_jets = np.zeros((n_events,max_n_jets,4))
py_events_particles = np.zeros((n_events,max_n_particles,3))

####################################
# Outer loop over events
####################################
for idx_event,_ in enumerate(range(n_events)):
   
    if not pythia.next():
        continue

    particles = []

    pidx=0        
    for idx,p in enumerate(pythia.event):
        if p.isFinal() and p.isHadron() and (abs(p.y()) < particle_y_cut) and (p.pT() > particle_pt_cut):                    
            
            pj = fj.PseudoJet( p.px(), p.py(), p.pz(), p.e() )
            particles.append( pj )
            
            if(pidx<max_n_particles):
                py_events_particles[idx_event, pidx, 0] = pj.pt()
                py_events_particles[idx_event, pidx, 1] = pj.eta()
                py_events_particles[idx_event, pidx, 2] = pj.phi()
            
            pidx+=1

    # Make the jets
    kt_cluster = fj.ClusterSequence(particles, kt_jetdef)
    akt_cluster = fj.ClusterSequence(particles, akt_jetdef)
    ca_cluster = fj.ClusterSequence(particles, ca_jetdef)

#############################################################
# GET THE JETS FROM EACH ALGO, SORTED BY pT
#############################################################
kt_jets = fj.sorted_by_pt(kt_cluster.inclusive_jets(min_jet_pt))
akt_jets = fj.sorted_by_pt(akt_cluster.inclusive_jets(min_jet_pt))
ca_jets = fj.sorted_by_pt(ca_cluster.inclusive_jets(min_jet_pt))

print("Clustered with "+kt_jetdef.description())
# print("The leading jet pT is: "+str(kt_jets[0].perp())+" GeV\n")

print("Clustered with "+akt_jetdef.description())
# print("The akt jet pT is: "+str(akt_jets[0].perp())+" GeV\n")

print("Clustered with "+ca_jetdef.description())
# print("The CA jet pT is: "+str(ca_jets[0].perp())+" GeV\n")

#############################################################
# CONSTITUENTS FOR THE LEADING JET, KT
#############################################################
print("kt jet info ... ");
print("idx\ty\t\tphi\t\tpt\t\tn constituents");

# print out the details for each jet
for i in range(0,1):
    # get the constituents of the jet
    constituents = fj.sorted_by_pt(kt_jets[i].constituents())
    print(i,"\t",
          "%0.4f"%kt_jets[i].rap(), "\t",
          "%0.4f"%kt_jets[i].phi(),"\t",
          "%0.4f"%kt_jets[i].perp(),"\t",
          len(constituents))
    print("\nConstituent info:       ");
    for j in range(0,len(constituents)) :
        print(j,"\t",
              "%0.4f"%constituents[j].rap(),"\t",
              "%0.4f"%constituents[j].phi(),"\t",
              "%0.4f"%constituents[j].perp(),"\t",
             );

    # plot the constituents
    zf = 2

    avg_rap = sum([c.rap() for c in constituents]) / len(constituents)
    avg_phi = sum([c.phi() for c in constituents]) / len(constituents)
    
    plt.scatter([c.rap()-avg_rap for c in constituents],
                [c.phi()-avg_phi for c in constituents],
                s=[c.perp()*zf for c in constituents],
                # s=[np.log(c.perp())*zf for c in constituents],
                color='red')
    plt.xlim(-0.4,0.4)
    plt.ylim(-0.4,0.4)
    plt.ylabel('Azimuth')
    plt.xlabel('Rapidity')
    plt.savefig('img-historytests/kth.jpg')
    plt.close()

#############################################################
# CONSTITUENTS FOR THE LEADING JET, AKT
#############################################################
print("akt jet info ... ");
print("idx\ty\t\tphi\t\tpt\t\tn constituents");
# print("        indices of constituents\n\n");

# print out the details for each jet
for i in range(0,1):
    # get the constituents of the jet
    constituents = fj.sorted_by_pt(akt_jets[i].constituents())

    print(i,"\t",
          "%0.4f"%akt_jets[i].rap(), "\t",
          "%0.4f"%akt_jets[i].phi(),"\t",
          "%0.4f"%akt_jets[i].perp(),"\t",
          len(constituents))

    print("\nConstituent info:       ");
    for j in range(0,len(constituents)) :
        print(j,"\t",
              "%0.4f"%constituents[j].rap(),"\t",
              "%0.4f"%constituents[j].phi(),"\t",
              "%0.4f"%constituents[j].perp(),"\t",
             );

    zf = 2
    avg_rap = sum([c.rap() for c in constituents]) / len(constituents)
    avg_phi = sum([c.phi() for c in constituents]) / len(constituents)
    
    plt.scatter([c.rap()-avg_rap for c in constituents],
                [c.phi()-avg_phi for c in constituents],
                s=[c.perp()*zf for c in constituents],
                # s=[np.log(c.perp())*zf for c in constituents],
                color='purple')
    plt.xlim(-0.4,0.4) # changing to +/-4.9 to show both events
    plt.ylim(-0.4,0.4)
    plt.ylabel('Azimuth')
    plt.xlabel('Rapidity')
    plt.savefig('img-historytests/akth.jpg')
    plt.close()

#############################################################
# CONSTITUENTS FOR THE LEADING JET, CA
#############################################################
print("ca jet info ... ");
print("idx\ty\t\tphi\t\tpt\t\tn constituents");
# print("        indices of constituents\n\n");

# print out the details for each jet
for i in range(0,1):
    # get the constituents of the jet
    constituents = fj.sorted_by_pt(ca_jets[i].constituents())

    print(i,"\t",
          "%0.4f"%ca_jets[i].rap(), "\t",
          "%0.4f"%ca_jets[i].phi(),"\t",
          "%0.4f"%ca_jets[i].perp(),"\t",
          len(constituents))

    print("\nConstituent info:       ");
    for j in range(0,len(constituents)) :
        print(j,"\t",
              "%0.4f"%constituents[j].rap(),"\t",
              "%0.4f"%constituents[j].phi(),"\t",
              "%0.4f"%constituents[j].perp(),"\t",
             );

    zf = 2
    avg_rap = sum([c.rap() for c in constituents]) / len(constituents)
    avg_phi = sum([c.phi() for c in constituents]) / len(constituents)
    
    plt.scatter([c.rap()-avg_rap for c in constituents],
                [c.phi()-avg_phi for c in constituents],
                s=[c.perp()*zf for c in constituents],
                # s=[np.log(c.perp())*zf for c in constituents],
                color='blue')
    # plt.xlim(-4.5,4.5)
    # plt.ylim(-np.pi,2*np.pi)
    plt.xlim(-0.4,0.4)
    plt.ylim(-0.4,0.4)
    plt.ylabel('Azimuth')
    plt.xlabel('Rapidity')
    plt.savefig('img-historytests/cah.jpg')
    plt.close()