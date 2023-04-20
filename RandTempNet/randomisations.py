#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------
#-        Temporal Networks v2.0          -
#-           by Mathieu GÉNOIS            -
#-       genois.mathieu@gmail.com         -
#-  adapted in python3 by Thomas Robiglio -
#-       robigliothomas@gmail.com         -
#------------------------------------------
#Python module for handling temporal networks
#------------------------------------------
#==========================================
#==========================================
#------------------------------------------
#Libraries
from .classes import *
from .utils import *
from . import utils as tnu
from . import measures as tnm
from random import choice,sample,randint,shuffle
from copy import deepcopy
import itertools as it
#import measures as tnm
#import utils as tnu
import networkx as nx
import numpy as np
#------------------------------------------
#==========================================
#==========================================
#------------------------------------------
#Randomization functions
#------------------------------------------
#==========================================
#------------------------------------------
#Basic shufflings
#------------------------------------------
#==========================================
#------------------------------------------
#P__1 (Global instant-event shuffling)
# > Breaks contacts into events and shuffles them across all possible links
#  seq_data (snapshot_sequence())
def P__1(seq_data):
    #construction of the new snapshot_sequence
    list_time = list(seq_data.data.keys())
    list_time.sort()
    t_i = list_time[0]
    t_f = list_time[-1]
    dt = list_time[1] - t_i
    t_f += dt
    Output = snapshot_sequence(t_i,t_f,dt)
    #extraction of all active nodes
    data = seq_data.out()
    nodes = list(set().union(*[step[1] for step in data]))
    nodes = list(set().union(*[lk.display() for lk in nodes]))
    #extraction of the number of events
    nE = sum([len(step[1]) for step in data])
    list_t = list(range(t_i,t_f,dt))
    #permutation
    for k in range(nE):
        t = choice(list_t)
        i,j = sample(nodes,2)
        while link(i,j) in Output.data[t].out():
            i,j = sample(nodes,2)
        Output.data[t].add_link(i,j)
    return Output
#------------------------------------------
#P__ptau (Global events shuffling)
#  lks_data (link_timeline())
#  ti (int): first time step of the dataset
#  tf (int): last time step of the dataset
#  dt: duration of a time step (int)
def P__ptau(lks_data,ti,tf,dt):
    #links extraction
    list_lk = list(lks_data.links())
    #nodes extraction
    nodes = list(set().union(*[lk.display() for lk in list_lk]))
    nN = len(nodes)
    index_node = {nodes[k]:k for k in range(nN)}
    #contacts extraction with time stamps
    list_c = list(it.chain(*list(lks_data.data.values())))
    #list_c = list((lks_data.data.values()))
    #contacts redistribution
    Output = link_timeline()
    Tl = {}
    for c in list_c:
        #choice of the link
        n,p = sample(nodes,2)
        lk = link(n,p)
        #virtual extension of the contact to test for concatenation
        t0 = randint(ti/dt,(tf - c.duration)/dt)*dt
        t1 = t0 + c.duration
        loc_c = list(range(t0-dt,t1+dt,dt)) #extended list of activation times
        #test for overlapping
        if lk in Tl:
            test = np.array([t in Tl[lk] for t in loc_c])
            while test.any():
                n,p = sample(nodes,2)
                lk = link(n,p)
                #virtual extension of the contact to test for concatenation
                t0 = randint(ti/dt,(tf - c.duration)/dt)*dt
                t1 = t0 + c.duration
                loc_c = list(range(t0-dt,t1+dt,dt)) #extended list of activation times
                if lk in Tl:
                    test = np.array([t in Tl[lk] for t in loc_c])
                else:
                    test = np.array([False])
            if lk in Tl:
                Output.add_contact(lk.i,lk.j,t0,c.duration)
#                Output.add_contact(lk.i,lk.j,c.time,c.duration)
            else:
                Tl[lk] = []
                Output.add_link(lk.i,lk.j,[(t0,c.duration)])
#                Output.add_link(lk.i,lk.j,[(c.time,c.duration)])
        else:
            Tl[lk] = []
            Output.add_link(lk.i,lk.j,[(t0,c.duration)])
#            Output.add_link(lk.i,lk.j,[(c.time,c.duration)])
        Tl[lk] += loc_c
    return Output
#------------------------------------------
#==========================================
#------------------------------------------
# Link shuffling
#------------------------------------------
#==========================================
#------------------------------------------
#P__pTheta (Permutation of links)
#lks_data: (link_timeline())
def P__pTheta(lks_data):
    list_links = lks_data.links()
    nLinks = len(list_links)
    #extraction of the timelines and the list of nodes
    list_timeline = [[c.display() for c in lks_data.data[lk]] for lk in list_links]
    #dictionary of node indices
    nodes = list(set().union(*[lk.display() for lk in list_links]))
    nN = len(nodes)
    index_node = {nodes[k]:k for k in range(nN)}
    #construction of the new links
    Output = link_timeline()
    #new adjacency matrix to avoid link duplicates and self-loops
    Adj = np.identity(nN)
    for l in range(nLinks):
        #sampling of two nodes
        (i,j) = sample(nodes,2)
        #loop while self-loop or link already exists
        while Adj[index_node[i]][index_node[j]]:
            (i,j) = sample(nodes,2)
        #sampling of the timelines
        Tl = choice(list_timeline)
        list_timeline.remove(Tl)
        #creation of the new link
        Output.add_link(i,j,Tl)
        #update of the adjacency matrix
        Adj[index_node[i]][index_node[j]] = 1
        Adj[index_node[j]][index_node[i]] = 1
    return Output
#------------------------------------------
#P__I_pTheta (Permutation of links with preservation of connectivity)
#lks_data: (link_timeline())
def P__I_pTheta(lks_data):
    not_connected = True
    while not_connected:
        Output = P__pTheta(lks_data)
        G = tnu.aggregate_link_timeline(Output)
        if nx.is_connected(G):
            not_connected = False
    return Output
#------------------------------------------
#P__pTheta_k (Permutation of links with degree preservation)
#  lks_data (link_timeline())
def P__k_pTheta(lks_data):
    list_links = lks_data.links()
    #extraction of the timelines
    list_timeline = [[c.display() for c in lks_data.data[lk]] for lk in list_links]
    #extraction of degrees and node list
    deg = tnm.degrees(list_links)
    nodes = list(deg.keys())
    nodes = sorted(nodes,key=lambda x:deg[x])
    nodes = [n for n in nodes if deg[n] > 0]
    #handling unsolvable finalisation of the reconstruction
    redo = True
    while redo:
        redo = False
        #link permutation
        Output = link_timeline()
        list_nodes = nodes[:] #list of available nodes
        list_tl = list_timeline[:]
        k_loc = deepcopy(deg)
        while list_nodes != []:
            n = list_nodes.pop()
            #test of solvable finalisation of the reconstruction
            if len(list_nodes) >= k_loc[n]:
                neighbors = sample(list_nodes,k_loc[n])
                k_loc[n] = 0
                for p in neighbors:
                    k_loc[p] -= 1
                    Tl = choice(list_tl)
                    list_tl.remove(Tl)
                    Output.add_link(n,p,Tl)
                #updating the list of available nodes
                list_nodes = [n for n in list_nodes if k_loc[n] > 0]
                list_nodes = sorted(list_nodes,key=lambda x:k_loc[x])
            else:
                list_nodes = []
                redo = True
    return Output
#------------------------------------------
#P__I_pTheta (Permutation of links with preservation of connectivity)
#lks_data: (link_timeline())
def P__k_I_pTheta(lks_data):
    not_connected = True
    while not_connected:
        Output = P__k_pTheta(lks_data)
        G = tnu.aggregate_link_timeline(Output)
        if nx.is_connected(G):
            not_connected = False
    return Output
#------------------------------------------
#P__pTheta_sigma_SigmaL (Permutation of links with group structure preservation)
#  lks_data (link_timeline()):
#  group: dictionary associating each node with its group. Group labels can be of any type.
def P__pTheta_sigma_SigmaL(lks_data,group):
    #extraction of the group labels
    group_labels = list(set(group.values()))
    #dictionary of node indices
    nodes = list(group.keys())
    nN = len(nodes)
    index_node = {nodes[k]:k for k in range(nN)}
    #initialisation of the list of nodes per group
    list_node = {lbl:[] for lbl in group_labels}
    #extraction of the half-links (as group affiliations), the timelines and the lists of nodes per group
    list_timeline = []
    list_group = []
    list_links = lks_data.links()
    nLinks = len(list_links)
    for lk in list_links:
        list_timeline.append([c.display() for c in lks_data.data[lk]])
        list_group.append((group[lk.i],group[lk.j]))
        list_node[group[lk.i]].append(lk.i)
        list_node[group[lk.j]].append(lk.j)
    for lbl in group_labels:
        list_node[lbl] = list(set(list_node[lbl]))
    #construction of the new links
    new_links = []
    #new adjacency matrix to avoid link duplicates and self-loops
    Adj = np.identity(nN)
    for n in range(nLinks):
        #sampling of two half-links
        (x,y) = choice(list_group)
        list_group.remove((x,y))
        i = choice(list_node[x])
        j = choice(list_node[y])
        #loop while self-loop or link already exists
        while Adj[index_node[i]][index_node[j]]:
            i = choice(list_node[x])
            j = choice(list_node[y])
        #creation of a new link
        new_links.append((i,j))
        #update of the adjacency matrix
        Adj[index_node[i]][index_node[j]] = 1
        Adj[index_node[j]][index_node[i]] = 1
    Output = link_timeline(new_links)
    #sampling of the timelines
    for i,j in new_links:
        Tl = choice(list_timeline)
        list_timeline.remove(Tl)
        Output.data[link(i,j)] = Tl[:]
    return Output
#------------------------------------------
#P__k_pTheta_sigma_SigmaL (Permutation of links with group structure and node degrees preservation)
#  lks_data: (link_timeline())
#  group: dictionary associating each node with its group. Group labels can be of any type.
#  n_iter: factor for the number of iterations: number of permutations = n_iter x number of links. By default, n_iter = 3
def P__k_pTheta_sigma_SigmaL(lks_data,group,n_iter=3):
    list_links = lks_data.links()
    Output = link_timeline(lks_data.links_display())
    nL = len(list_links)
    #initialisation of the convergence tracking
    converge = [1]
    #extraction of the group labels
    group_labels = list(set(group.values()))
    nG = len(group_labels)
    #extraction of the nodes
    nodes = list(group.keys())
    nN = len(nodes)
    #dictionary of node indices
    node_index = {n:nodes.index(n) for n in nodes}
    #computation of the adjacency matrix and extraction of the links by group pair
    A = np.identity(nN)
    lk_g = {}
    for x in range(nG):
        for y in range(x+1):
            lk_g[(x,y)] = []
    for lk in list_links:
        z_i = node_index[lk.i]
        z_j = node_index[lk.j]
        A[z_i][z_j] = 1
        A[z_j][z_i] = 1
        x_i = group_labels.index(group[lk.i])
        x_j = group_labels.index(group[lk.j])
        x = max(x_i,x_j)
        y = min(x_i,x_j)
        lk_g[(x,y)].append(lk)
    #extraction of the timelines
    list_timeline = [[c.display() for c in lks_data.data[lk]] for lk in list_links]
    #copy of the original adjacency matrix
    A_init = deepcopy(A.reshape(-1))
    #permutations
    for iteration in range(n_iter*nL):
        #choice of the first link
        list_lk = list(Output.links())
        lk1 = choice(list_lk)
        #determination of the group pair
        x_i = group_labels.index(group[lk1.i])
        x_j = group_labels.index(group[lk1.j])
        x = max(x_i,x_j)
        y = min(x_i,x_j)
        #choice of the second link
        pool = lk_g[(x,y)][:]
        pool.remove(lk1)
        #case where there is only one link between the two groups
        if pool == []:
            lk2 = lk1
        else:
            lk2 = choice(pool)
        #determination of the nodes from the same group
        if group[lk1.i] == group[lk2.i]:
            n1,n2 = lk1.i,lk2.i
            p1,p2 = lk2.j,lk1.j
        else:
            n1,n2 = lk1.i,lk2.j
            p1,p2 = lk2.i,lk1.j
        #test of existence of the future links
        while A[node_index[n1]][node_index[p1]] or A[node_index[n2]][node_index[p2]]:
            #choice of the first link
            lk1 = choice(list(Output.links()))
            #determination of the pair of groups
            x_i = group_labels.index(group[lk1.i])
            x_j = group_labels.index(group[lk1.j])
            x = max(x_i,x_j)
            y = min(x_i,x_j)
            #choice of the second link
            pool = lk_g[(x,y)][:]
            pool.remove(lk1)
            #case where there is only one link between the two groups
            if pool == []:
                lk2 = lk1
            else:
                lk2 = choice(pool)
            #determination of the nodes from the same group
            if group[lk1.i] == group[lk2.i]:
                n1,n2 = lk1.i,lk2.i
                p1,p2 = lk2.j,lk1.j
            else:
                n1,n2 = lk1.i,lk2.j
                p1,p2 = lk2.i,lk1.j
        #suppressing the former links
        Output.del_link(lk1.i,lk1.j)
        Output.del_link(lk2.i,lk2.j)
        A[node_index[lk1.i]][node_index[lk1.j]] = 0
        A[node_index[lk1.j]][node_index[lk1.i]] = 0
        A[node_index[lk2.i]][node_index[lk2.j]] = 0
        A[node_index[lk2.j]][node_index[lk2.i]] = 0
        lk_g[(x,y)].remove(lk1)
        lk_g[(x,y)].remove(lk2)
        #adding the new links
        lk1 = link(n1,p1)
        lk2 = link(n2,p2)
        Output.add_link(lk1.i,lk1.j)
        Output.add_link(lk2.i,lk2.j)
        A[node_index[lk1.i]][node_index[lk1.j]] = 1
        A[node_index[lk1.j]][node_index[lk1.i]] = 1
        A[node_index[lk2.i]][node_index[lk2.j]] = 1
        A[node_index[lk2.j]][node_index[lk2.i]] = 1
        lk_g[(x,y)].append(lk1)
        lk_g[(x,y)].append(lk2)
        #calculation of the jaccard index to follow convergence
        inter = list(set(list_links) & set(Output.links()))
        union = list(set(list_links) | set(Output.links()))
        converge.append(float(len(inter))/float(len(union)))
    #timelines redistribution
    for lk in Output.links():
        Tl = list_timeline.pop()
        Output.add_link(lk.i,lk.j,Tl)
    return Output,converge
#------------------------------------------
#P__G_psigma (Permutation of group labels)
#  group: dictionary associating each node with its group. Group labels can be of any type.
def P__G_psigma(group):
    #extraction of the group labels
    group_labels = list(group.values())
    #random attribution of group labels
    nodes = list(group.keys())
    nN = len(nodes)
    shuffle(group_labels)
    new_labels = {nodes[i]:group_labels[i] for i in range(nN)}
    return new_labels
#------------------------------------------
#==========================================
#------------------------------------------
# Timeline shuffling - instant-events
#------------------------------------------
#==========================================
#------------------------------------------
#P__L (Global events permutation)
#  lks_data (link_timeline())
#  ti (int): first time step of the dataset
#  tf (int): last time step of the dataset
#  dt (int): duration of a time step
def P__L(lks_data,ti,tf,dt):
    #extraction of the total number of contacts
    nC = sum(tnm.weights(lks_data).values())
    #extraction of the links
    list_lk = list(lks_data.links())
    #events redistribution
    Output = tij()
    for e in range(nC):
        t = randint(ti/dt,tf/dt)*dt
        lk = choice(list_lk)
        Output.add_event(t,lk.i,lk.j)
    return Output
#------------------------------------------
#P__w (Global events permutation with weights preservation)
#  lks_data (link_timeline())
#  ti (int): first time step of the dataset
#  tf (int): last time step of the dataset
#  dt (int): duration of a time step
def P__w(lks_data,ti,tf,dt):
    #events redistribution
    Output = tij()
    #link weights extraction
    weight = tnm.weights(lks_data)
    for lk in lks_data.links():
        w = weight[lk]
        list_t = sample(list(range(ti,tf,dt)),w)
        for t in list_t:
            Output.add_event(t,lk.i,lk.j)
    return Output
#------------------------------------------
#P__w_t1_tw (Global events permutation with weights, initial time and final time preservation)
#  lks_data (link_timeline())
#  ti (int): first time step of the dataset
#  tf (int): last time step of the dataset
#  dt (int): duration of a time step
def P__w_t1_tw(lks_data,ti,tf,dt):
    #events redistribution
    Output = tij()
    #link weights extraction
    weight = tnm.weights(lks_data)
    for lk in lks_data.links():
        w = weight[lk]
        Tl = sorted(lks_data.data[lk],key=lambda x:x.time)
        t1 = Tl[0].time
        tw = Tl[-1].time + Tl[-1].duration - 1
        #adding initial event
        Output.add_event(t1,lk.i,lk.j)
        #adding final event
        if tw > t1:
            Output.add_event(tw,lk.i,lk.j)
        #adding other events
        if w > 2:
            list_t = sample(list(range(t1+1,tw,dt)),w-2)
            for t in list_t:
                Output.add_event(t,lk.i,lk.j)
    return Output
#------------------------------------------
#==========================================
#------------------------------------------
# Timeline shuffling - events
#------------------------------------------
#==========================================
#------------------------------------------
#P__L_ptau (Global contacts shuffling)
#  lks_data (link_timeline())
#  ti (int): first time step of the dataset
#  tf (int): last time step of the dataset
#  dt: duration of a time step (int)
def P__L_ptau(lks_data,ti,tf,dt):
    #links extraction
    list_lk = list(lks_data.links())
    #contacts extraction with time stamps
    list_c = list(it.chain(*list(lks_data.data.values())))
    #contacts redistribution
    Output = link_timeline(lks_data.links_display())
    Tl = {lk:[] for lk in list_lk}
    for c in list_c:
        #choice of the link
        lk = choice(list_lk)
        #virtual extension of the contact to test for concatenation
        t0 = randint(ti/dt,(tf - c.duration)/dt)*dt
        t1 = t0 + c.duration
        loc_c = list(range(t0-dt,t1+dt,dt)) #extended list of activation times
        #test for overlapping
        test = np.array([t in Tl[lk] for t in loc_c])
        while test.any(): 
            lk = choice(list_lk)
            t0 = randint(ti/dt,(tf - c.duration)/dt)*dt
            t1 = t0 + c.duration
            loc_c = list(range(t0-dt,t1+dt,dt)) #extended list of activation times
            test = np.array([t in Tl[lk] for t in loc_c])
        Output.add_contact(lk.i,lk.j,t0,c.duration)
        Tl[lk] += loc_c
    return Output
#------------------------------------------
#==========================================
#------------------------------------------
# Timeline shuffling (events & contacts)
#------------------------------------------
#==========================================
#------------------------------------------
#P__pitau (Contacts shuffling in place)
#  lks_data (link_timeline())
#  ti (int): first time step of the dataset
#  tf (int): last time step of the dataset
#  dt (int): duration of a time step
def P__pitau(lks_data,ti,tf,dt):
    Output = link_timeline()
    dict_tau = tnm.contact_durations(lks_data)
    for lk in lks_data.links():
        #extraction of the contact events
        contacts = dict_tau[lk]
        #computation of the remaining usable time
        tu = tf - ti - sum(contacts)
        #construction of the new timeline
        contacts = sample(contacts,len(contacts))
        #new starting times for the contacts
        list_t = [randint(0,tu/dt)*dt for c in contacts]
        list_t.sort()
        delta = 0
        Tl = []
        for c in contacts:
            t0 = list_t.pop(0)
            t0 += delta
            Tl.append((t0,c))
            delta += c
        #adding the timeline
        Output.add_link(lk.i,lk.j,Tl)
    return Output
#------------------------------------------
#P__pitau_t1_tw (Contacts shuffling in place)
#  lks_data (link_timeline())
#  dt (int): duration of a time step
def P__pitau_t1_tw(lks_data,dt):
    Output = link_timeline()
    dict_tau = tnm.contact_durations(lks_data)
    for lk in lks_data.links():
        #extraction of the contact events
        contacts = dict_tau[lk]
        nC = len(contacts)
        #parameters
        Tl = sorted(lks_data.data[lk],key=lambda x:x.time)
#        if nC == 2:
#            print Tl
        t1 = Tl[0].time
        tw = Tl[-1].time + Tl[-1].duration
        tu = tw - t1 - sum(contacts)
        #construction of the new timeline
        contacts = sample(contacts,len(contacts))
        #first contact
        c1 = contacts.pop()
        Tl = [(t1,c1)]
        if nC > 1:
            #last contact
            cw = contacts.pop()
            Tl.append((tw-cw,cw))
            #other contacts
            list_t = [randint(1,tu/dt)*dt for c in contacts]
#            list_t = sample(np.arange(1,tu/dt,1)*dt,nC-2)
            list_t.sort()
            delta = t1+c1
            for c in contacts:
                t0 = list_t.pop(0)
                t0 += delta
                Tl.append((t0,c))
                delta += c
#        if nC == 2:
#            print Tl
#            raw_input()
        #adding the timeline
        Output.add_link(lk.i,lk.j,Tl)
    return Output
#------------------------------------------
#P__pitau_pidtau (Interval shuffling in place)
#  lks_data (link_timeline())
#  ti (int): first time step of the dataset
#  tf (int): last time step of the dataset
#  dt (int): duration of a time step
def P__pitau_pidtau(lks_data,ti,tf,dt):
    Output = link_timeline()
    dict_tau = tnm.contact_durations(lks_data)
    dict_dtau = tnm.intercontact_durations(lks_data)
    for lk in lks_data.links():
        #extraction of the contact and intercontact events
        contacts = dict_tau[lk]
        intercontacts = dict_dtau[lk]
        #computation of the remaining usable time
        tu = tf - ti - sum(contacts) - sum(intercontacts)
        #new initial time
        t0 = randint(0,tu/dt)*dt
        #construction of the new timeline
        contacts = sample(contacts,len(contacts))
        intercontacts = sample(intercontacts,len(intercontacts))
        t = t0
        Tl = []
        while contacts != []:
            tau = contacts.pop()
            Tl.append((t,tau))
            t += tau
            if intercontacts != []:
                dtau = intercontacts.pop()
                t += dtau
        #adding the timeline
        Output.add_link(lk.i,lk.j,Tl)
    return Output
#------------------------------------------
#P__pitau_pidtau_t1 (Interval shuffling in place with conservation of the initial time)
#  lks_data (link_timeline())
def P__pitau_pidtau_t1(lks_data):
    Output = link_timeline()
    dict_tau = tnm.contact_durations(lks_data)
    dict_dtau = tnm.intercontact_durations(lks_data)
    for lk in lks_data.links():
        #extraction of the contact and intercontact events
        contacts = dict_tau[lk]
        intercontacts = dict_dtau[lk]
        #initial time
        list_c = lks_data.data[lk]
        list_c = sorted(list_c,key=lambda x:x.time)
        t0 = list_c[0].time
        #construction of the new timeline
        contacts = sample(contacts,len(contacts))
        intercontacts = sample(intercontacts,len(intercontacts))
        t = t0
        Tl = []
        while contacts != []:
            tau = contacts.pop()
            Tl.append((t,tau))
            t += tau
            if intercontacts != []:
                dtau = intercontacts.pop()
                t += dtau
        #adding the timeline
        Output.add_link(lk.i,lk.j,Tl)
    return Output
#------------------------------------------
#P__perTheta (Periodic boudaries offset shuffling)
#  lks_data (link_timeline())
#  ti (int): first time step of the dataset
#  tf (int): last time step of the dataset
#  dt (int): duration of a time step
def P__perTheta(lks_data,ti,tf,dt):
    Output = link_timeline()
    tu = tf - ti
    for lk in lks_data.links():
        Tl = sorted(lks_data.data[lk],key=lambda x:x.time)
        t2 = Tl[-1].time + Tl[-1].duration
        offset = randint(0,tu/dt)*dt
        new_Tl = []
        for c in Tl:
            t1 = c.time + offset
            t2 = t1 + c.duration
            if (t1 < tf)and(t2 > tf):
                tau1 = tf - t1
                tau2 = t2 - tf
                new_Tl.append((t1,tau1))
                new_Tl.append((ti,tau2))
            else:
                new_Tl.append((int((c.time - ti + offset)%tu) + ti,c.duration))
        Output.add_link(lk.i,lk.j,new_Tl)
    return Output
#------------------------------------------
#P__tau_dtau (Offset shuffling)
#  lks_data (link_timeline())
#  ti (int): first time step of the dataset
#  tf (int): last time step of the dataset
#  dt (int): duration of a time step
def P__tau_dtau(lks_data,ti,tf,dt):
    Output = link_timeline()
    for lk in lks_data.links():
        Tl = sorted(lks_data.data[lk],key=lambda x:x.time)
        t1 = Tl[0].time
        t2 = Tl[-1].time + Tl[-1].duration
        tu = tf - t2 + t1 - ti #unused time
        offset = randint(0,tu/dt)*dt + ti - t1
        new_Tl = []
        for c in Tl:
            new_Tl.append((c.time + offset,c.duration))
        Output.add_link(lk.i,lk.j,new_Tl)
    return Output
#------------------------------------------
#==========================================
#------------------------------------------
#Snapshot shuffling
#------------------------------------------
#==========================================
#------------------------------------------
#P__t
# > Permutation of links in each snapshot
#  seq_data (snapshot_sequence())
def P__t(seq_data):
    #construction of the new snapshot_sequence
    list_time = list(seq_data.data.keys())
    list_time.sort()
    t_i = list_time[0]
    t_f = list_time[-1]
    dt = list_time[1] - t_i
    t_f += dt
    Output = snapshot_sequence(t_i,t_f,dt)
    #extraction of all active nodes
    nodes = list(set().union(*[step[1] for step in seq_data.out()]))
    nodes = list(set().union(*[lk.display() for lk in nodes]))
    #permutation
    for t,list_link in seq_data.out():
        if list_link != []:
            new_list = []
            #permutation
            for lk in list_link:
                i,j = sample(nodes,2)
                while link(i,j) in new_list:
                    i,j = sample(nodes,2)
                new_list.append(link(i,j))
            Output.update_snapshot(t,new_list)
    return Output
#------------------------------------------
#P__t_Phi
# > Permutation of links between active nodes in each snapshot
#  seq_data (snapshot_sequence())
def P__t_Phi(seq_data):
    #construction of the new snapshot_sequence
    list_time = list(seq_data.data.keys())
    list_time.sort()
    t_i = list_time[0]
    t_f = list_time[-1]
    dt = list_time[1] - t_i
    t_f += dt
    Output = snapshot_sequence(t_i,t_f,dt)
    #permutation
    for t,list_link in seq_data.out():
        if list_link != []:
            #list of active nodes in the snapshot
            nodes = list(set().union(*[lk.display() for lk in list_link]))
            new_list = []
            #permutation
            for lk in list_link:
                i,j = sample(nodes,2)
                while link(i,j) in new_list:
                    i,j = sample(nodes,2)
                new_list.append(link(i,j))
            Output.update_snapshot(t,new_list)
    return Output
#------------------------------------------
#P__d
# > Permutation of links with degree preservation in each snapshot
#  seq_data (snapshot_sequence())
#  link_threshold (int): minimum number of links to use the configuration model algorithm
#  n_iter (int): parameter for the number of iterations in the case of the Sneppen-Maslov algorithm
def P__d(seq_data,link_threshold=20,n_iter=5):
    #extraction of snapshots to shuffle
    todo = [s for s in seq_data.out() if len(s[1]) > 1]
    for t,list_link in todo:
        #configuration model
        if len(list_link) >= link_threshold:
            #extraction of degrees and node list
            deg = degrees(list_link)
            nodes = list(deg.keys())
            nodes = sorted(nodes,key=lambda x:deg[x])
            nodes = [n for n in nodes if deg[n] > 0]
            list_lk = []
            #handling unsolvable finalisations of the reconstruction
            redo = True
            while redo:
                redo = False
                #link permutation
                list_nodes = nodes[:]
                k_loc = deepcopy(deg)
                while list_nodes != []:
                    n = list_nodes.pop()
                    #test of solvable finalisation of the reconstruction
                    if len(list_nodes) >= deg[n]:
                        neighbors = sample(list_nodes,deg[n])
                        deg[n] = 0
                        for p in neighbors:
                            deg[p] -= 1
                            list_lk.append((n,p))
                        #updating the list of nodes
                        list_nodes = [n for n in list_nodes if deg[n] > 0]
                        list_nodes = sorted(list_nodes,key=lambda x:deg[x])
                    else:
                        redo = True
                        list_nodes = []
            seq_data.update_snapshot(t,list_lk)
        #sneppen-maslov
        else:
            nL = len(list_link)
            for z in range(n_iter*nL):
                redo = True
                while redo:
                    redo = False
                    l1 = choice(list_link)
                    l2 = choice(list_link)
                    n1,p1 = l1.display()
                    n2,p2 = l2.display()
                    #test of common node
                    if len(set([n1,p1,n2,p2])) < 4:
                        redo = True
                    #tests of links already present
                    if link(n1,p2) in list_link:
                        redo = True
                    if link(n2,p1) in list_link:
                        redo = True
                seq_data.data[t].del_links([(n1,p1),(n2,p2)])
                seq_data.data[t].add_links([(n1,p2),(n2,p1)])
    return seq_data
#------------------------------------------
#P__isoGamma
# > Permutation of node identities in each snapshot
#  seq_data (snapshot_sequence())
def P__isoGamma(seq_data):
    #construction of the new snapshot_sequence
    list_time = list(seq_data.data.keys())
    list_time.sort()
    t_i = list_time[0]
    t_f = list_time[-1]
    dt = list_time[1] - t_i
    t_f += dt
    Output = snapshot_sequence(t_i,t_f,dt)
    #extraction of all active nodes
    nodes = list(set().union(*[step[1] for step in seq_data.out()]))
    nodes = list(set().union(*[lk.display() for lk in nodes]))
    #permutation
    for t,list_link in seq_data.out():
        if list_link != []:
            #extraction of active nodes
            old_nodes = list(set().union(*[lk.display() for lk in list_link]))
            nN = len(old_nodes)
            #definition of the mapping from old to new IDs
            new_nodes = sample(nodes,nN)
            transform = {old_nodes[k]:new_nodes[k] for k in range(nN)}
            #permutation
            new_list = []
            for lk in list_link:
                new_list.append(link(transform[lk.i],transform[lk.j]))
            Output.update_snapshot(t,new_list)
    return Output
#------------------------------------------
#P__isoGamma_Phi
# > Permutation of active node identities in snapshots
#  seq_data (snapshot_sequence())
def P__isoGamma_Phi(seq_data):
    #construction of the new snapshot_sequence
    list_time = list(seq_data.data.keys())
    list_time.sort()
    t_i = list_time[0]
    t_f = list_time[-1]
    dt = list_time[1] - t_i
    t_f += dt
    Output = snapshot_sequence(t_i,t_f,dt)
    #permutation
    for t,list_link in seq_data.out():
        if list_link != []:
            #extraction of active nodes
            old_nodes = list(set().union(*[lk.display() for lk in list_link]))
            nN = len(old_nodes)
            #definition of the mapping from old to new IDs
            new_nodes = sample(old_nodes,nN)
            transform = {old_nodes[k]:new_nodes[k] for k in range(nN)}
            #permutation
            new_list = []
            for lk in list_link:
                new_list.append(link(transform[lk.i],transform[lk.j]))
            Output.update_snapshot(t,new_list)
    return Output
#------------------------------------------
#P__pttau (Global contacts shuffling)
#  lks_data (link_timeline())
#  dt: duration of a time step (int)
def P__pttau(lks_data,dt):
    #links extraction
    list_lk = list(lks_data.links())
    #nodes extraction
    nodes = list(set().union(*[lk.display() for lk in list_lk]))
    nN = len(nodes)
    index_node = {nodes[k]:k for k in range(nN)}
    #contacts extraction with time stamps
    list_c = list(it.chain(*list(lks_data.data.values())))
    #contacts redistribution
    Output = link_timeline()
    Tl = {}
    for c in list_c:
        #choice of the link
        n,p = sample(nodes,2)
        lk = link(n,p)
        #virtual extension of the contact to test for concatenation
        ti = c.time
        tf = c.time + c.duration
        loc_c = list(range(ti-dt,tf+dt,dt)) #extended list of activation times
        #test for overlapping
        if lk in Tl:
            test = np.array([t in Tl[lk] for t in loc_c])
            while test.any():
                n,p = sample(nodes,2)
                lk = link(n,p)
                ti = c.time
                tf = c.time + c.duration
                loc_c = list(range(ti-dt,tf+dt,dt)) #extended list of activation times
                if lk in Tl:
                    test = np.array([t in Tl[lk] for t in loc_c])
                else:
                    test = np.array([False])
            if lk in Tl:
                Output.add_contact(lk.i,lk.j,c.time,c.duration)
            else:
                Tl[lk] = []
                Output.add_link(lk.i,lk.j,[(c.time,c.duration)])
        else:
            Tl[lk] = []
            Output.add_link(lk.i,lk.j,[(c.time,c.duration)])
        Tl[lk] += loc_c
    return Output
#------------------------------------------
#==========================================
#------------------------------------------
#Sequence shuffling
#------------------------------------------
#==========================================
#------------------------------------------
#P__pGamma (Permutation of snapshots)
#  seq_data (snapshot_sequence())
def P__pGamma(seq_data):
    #extraction of snapshot_sequence caracteristics
    list_time = list(seq_data.data.keys())
    list_time.sort()
    t_i = list_time[0]
    t_f = list_time[-1]
    dt = list_time[1] - t_i
    t_f += dt
    #snapshots treatment
    list_S = []
    #extraction of the active snapshots
    for t,list_link in seq_data.out():
        if list_link != []:
            list_S.append(list_link)
    #definition of the new time steps
    list_t = sample(list(range(t_i,t_f,dt)),len(list_S))
    #reconstruction
    Output = snapshot_sequence(t_i,t_f,dt)
    for t in list_t:
        list_link = list_S.pop()
        Output.update_snapshot(t,list_link)
    return Output
#------------------------------------------
#P__pGamma_sgnA (Permutation of active snapshots)
#  seq_data (snapshot_sequence())
def P__pGamma_sgnA(seq_data):
    #extraction of snapshot_sequence caracteristics
    list_time = list(seq_data.data.keys())
    list_time.sort()
    t_i = list_time[0]
    t_f = list_time[-1]
    dt = list_time[1] - t_i
    t_f += dt
    #snapshots treatment
    list_S = []
    #extraction of the active snapshots and their time stamps
    list_t = []
    for t,list_link in seq_data.out():
        if list_link != []:
            list_t.append(t)
            list_S.append(list_link)
    #definition of the new time steps
    list_t = sample(list_t,len(list_t))
    #reconstruction
    Output = snapshot_sequence(t_i,t_f,dt)
    for t in list_t:
        list_link = list_S.pop()
        Output.update_snapshot(t,list_link)
    return Output
#------------------------------------------
#==========================================
#------------------------------------------
#Intersections
#------------------------------------------
#==========================================
#------------------------------------------
#P__L_t (Global events shuffling)
#  lks_data (link_timeline())
#  dt (int): duration of a time step
def P__L_t(lks_data,dt):
    #events extraction
    list_t = []
    for lk in list_lk:
        for c in lks_data.data[lk]:
            ti = c.time
            tf = c.time + c.duration
            list_t += list(range(ti,tf,dt))
    #events redistribution
    Output = tij()
    for lk in list_lk:
        w = weight[lk]
        pool = list(set(list_t))
        list_event = sample(pool,w)
        for t in list_event:
            list_t.remove(t)
            Output.add_event(t,lk.i,lk.j)
    return Output
#------------------------------------------
#P__w_t (Global events shuffling)
#  lks_data (link_timeline())
#  dt (int): duration of a time step
def P__w_t(lks_data,dt):
    #link weights extraction
    weight = tnm.weights(lks_data)
    #link ordering by weight
    list_lk = sorted(lks_data.links(),key=lambda x:weight[x],reverse=True)
    #events extraction
    list_t = []
    for lk in list_lk:
        for c in lks_data.data[lk]:
            ti = c.time
            tf = c.time + c.duration
            list_t += list(range(ti,tf,dt))
    #events redistribution
    Output = tij()
    for lk in list_lk:
        w = weight[lk]
        pool = list(set(list_t))
        list_event = sample(pool,w)
        for t in list_event:
            list_t.remove(t)
            Output.add_event(t,lk.i,lk.j)
    return Output
#------------------------------------------
#P__L_pttau (Global contacts shuffling on Gstat)
#  lks_data (link_timeline())
#  dt: duration of a time step (int)
def P__L_pttau(lks_data,dt):
    #links extraction
    list_lk = list(lks_data.links())
    #contacts extraction with time stamps
    list_c = list(it.chain(*list(lks_data.data.values())))
    #contacts redistribution
    Output = link_timeline(lks_data.links_display())
    Tl = {lk:[] for lk in list_lk}
    for c in list_c:
        lk = choice(list_lk)
        #virtual extension of the contact to test for concatenation
        ti = c.time
        tf = c.time + c.duration
        loc_c = list(range(ti-dt,tf+dt,dt)) #extended list of activation times
        #test for overlapping
        test = np.array([t in Tl[lk] for t in loc_c])
        while test.any():
            lk = choice(list_lk)
            ti = c.time
            tf = c.time + c.duration
            loc_c = list(range(ti-dt,tf+dt,dt)) #extended list of activation times
            test = np.array([t in Tl[lk] for t in loc_c])
        Output.add_contact(lk.i,lk.j,c.time,c.duration)
        Tl[lk] += loc_c
    #Removal of links with empty timelines
    for lk in list_lk:
        if not Output.data[lk]:
            Output.del_link(lk.i,lk.j)
    return Output
#------------------------------------------
#P__n_pttau (Global contacts shuffling)
#  lks_data (link_timeline())
#  dt (int): duration of a time step
def P__n_pttau(lks_data,dt):
    #determination of the number of contacts
    num = tnm.number_of_contacts(lks_data)
    #link ordering by number of contact
    list_lk = sorted(lks_data.links(),key=lambda x:num[x],reverse=True)
    #contacts extraction
    list_c = list(it.chain(*list(lks_data.data.values())))
    #contacts redistribution
    Output = link_timeline()
    for lk in list_lk:
        #construction of a non-overlapping, non concatenating sample of n contacts
        n = num[lk]
        Tl = []
        #first contact
        c = choice(list_c)
        ti = c.time
        tf = c.time + c.duration
        loc_c = list(range(ti-dt,tf+dt,dt)) #extended list of activation times
        list_c.remove(c)
        Tl.append(c.display())
        testlist_t = loc_c[:] #test list for overlapping
        for ic in range(n-1):
            c = choice(list_c)
            ti = c.time
            tf = c.time + c.duration
            loc_c = list(range(ti-dt,tf+dt,dt))
            #test for overlapping and concatenation
            test = testlist_t + loc_c
            while len(test) > len(list(set(test))):
                c = choice(list_c)
                ti = c.time
                tf = c.time + c.duration
                loc_c = list(range(ti-dt,tf+dt,dt))
                test = testlist_t + loc_c
            #adding the contact
            testlist_t += loc_c
            list_c.remove(c)
            Tl.append(c.display())
        #adding the new timeline
        Output.add_link(lk.i,lk.j,Tl)
    return Output
#------------------------------------------
#P__L_pTheta (Permutation of timelines)
#lks_data: (link_timeline())
def P__L_pTheta(lks_data):
    #extraction of the timelines
    list_timeline = [[c.display() for c in lks_data.data[lk]] for lk in lks_data.links()]
    #construction of the new links
    Output = link_timeline()
    #sampling of the timelines
    for lk in lks_data.links():
        Tl = choice(list_timeline)
        list_timeline.remove(Tl)
        Output.add_link(lk.i,lk.j,Tl)
    return Output
#------------------------------------------
#P__w_pTheta (Permutation of timelines with preservation of the weights)
#lks_data (link_timeline())
def P__w_pTheta(lks_data):
    #extraction of the contact frequencies
    dict_link = tnm.weights(lks_data)
    #extraction of the timelines
    w_keys = list(set(dict_link.values()))
    dict_weight = {w:[] for w in w_keys}
    list_links = lks_data.links()
    for lk in list_links:
        dict_weight[dict_link[lk]].append([c.display() for c in lks_data.data[lk]])
    #construction of the new links
    Output = link_timeline()
    #sampling of the timelines
    for lk in list_links:
        w = dict_link[lk]
        Tl = choice(dict_weight[w])
        dict_weight[w].remove(Tl)
        Output.add_link(lk.i,lk.j,Tl)
    return Output
#------------------------------------------
#P__n_pTheta (Permutation of timelines with preservation of the contact frequencies)
#  lks_data (link_timeline())
def P__n_pTheta(lks_data):
    #extraction of the contact frequencies
    dict_link = tnm.number_of_contacts(lks_data)
    #extraction of the timelines
    n_keys = list(set(dict_link.values()))
    dict_timeline = {n:[] for n in n_keys}
    list_links = lks_data.links()
    for lk in list_links:
        dict_timeline[dict_link[lk]].append([c.display() for c in lks_data.data[lk]])
    #construction of the new links
    Output = link_timeline()
    #sampling of the timelines
    for lk in list_links:
        n = dict_link[lk]
        Tl = choice(dict_timeline[n])
        dict_timeline[n].remove(Tl)
        Output.add_link(lk.i,lk.j,Tl)
    return Output
#------------------------------------------
#------------------------------------------
