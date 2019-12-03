#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------
#-        Temporal Networks v1.0          -
#-           by Mathieu GÉNOIS            -
#-       genois.mathieu@gmail.com         -
#------------------------------------------
#Python module for handling temporal networks
#------------------------------------------
#==========================================
#==========================================
#------------------------------------------
#Libraries
from classes import *
from utils import *
from math import ceil
import numpy as np
import itertools as it
import matplotlib.pyplot as plt
#------------------------------------------
#==========================================
#==========================================
#------------------------------------------
#Measures
#------------------------------------------
#Computation of the degrees of the nodes
#-list_lk: list of link() objects
#>returns a dictionary {node:degree}
def degrees(list_lk):
    list_occurence = []
    for lk in list_lk:
        list_occurence.append(lk.i)
        list_occurence.append(lk.j)
    list_node = list(set(list_occurence))
    Output = {n:list_occurence.count(n) for n in list_node}
    return Output
#------------------------------------------
#Computation of the link contact matrix
#-list_lk: list of link() objects
#-group: dictionary of group affiliations
#>returns:
#  >contact matrix as numpy.array()
#  >list of group labels
def LCM(list_lk,group):
    #construction of the groups of nodes
    group_label = list(set(group.values()))
    group_count = [group.values().count(lbl) for lbl in group_label]
    nG = len(group_label)
    Output = np.zeros((nG,nG))
    for lk in list_lk:
        x = group_label.index(group[lk.i])
        y = group_label.index(group[lk.j])
        Output[x][y] += 1.
        Output[y][x] += 1.
    for n in range(nG):
        Output[n][n] /= 2.
        for p in range(nG):
            if n == p:
                norm = group_count[n]*(group_count[n]-1)/2.
            else:
                norm = group_count[n]*group_count[p]
            if norm > 0:
                Output[n][p] /= norm
    return Output,group_label
#------------------------------------------
#Computation of the weighted contact matrix
#-list_lk: list of link() objects
#-weights: dictionary of weights with link() objects as keys
#-group: dictionary of group affiliations
#>returns:
#  >contact matrix as numpy.array()
#  >list of group labels
def WCM(list_lk,weights,group):
    #construction of the groups of nodes
    group_label = list(set(group.values()))
    group_count = [group.values().count(lbl) for lbl in group_label]
    nG = len(group_label)
    Output = np.zeros((nG,nG))
    Norm = np.zeros((nG,nG))
    for lk in list_lk:
        x = group_label.index(group[lk.i])
        y = group_label.index(group[lk.j])
        w = weights[lk]
        Output[x][y] += w
        Output[y][x] += w
        Norm[x][y] += 1.
        Norm[y][x] += 1.
    for n in range(nG):
        Output[n][n] /= 2.
        for p in range(nG):
            if norm > 0:
                Output[n][p] /= Norm[n][p]
    return Output,group_label
#------------------------------------------
#Computation of the number of contacts for each link
#-lks_data (link_timeline()): object to analyse
#>returns a dictionary {link:number of contacts}
def number_of_contacts(lks_data):
    Output = {lk:len(lks_data.data[lk]) for lk in lks_data.links()}
    return Output
#------------------------------------------
#Computation of the weights for each link
#  lks_data (link_timeline()): object to analyse
def weights(lks_data):
    Output = {lk:sum([c.duration for c in lks_data.data[lk]]) for lk in lks_data.links()}
    return Output
#------------------------------------------
#Computation of the contact activity timeline
#  seq_data (snapshot_sequence()): object to analyse
def activity_timeline(seq_data):
    Output = [len(step[1]) for step in seq_data.out()]
    return Output
#------------------------------------------
#Computation of the contact durations
#  lks_data (link_timeline()): object to analyse
def contact_durations(lks_data):
    Output = {lk:[c.duration for c in lks_data.data[lk]] for lk in lks_data.links()}
    return Output
#------------------------------------------
#Computation of the intercontact durations list
#  lks_data (link_timeline()): object to analyse
def intercontact_durations(lks_data):
    Output = {}
    for lk in lks_data.links():
        td = [c.time for c in lks_data.data[lk]]
        tf = [c.time + c.duration for c in lks_data.data[lk]]
        values = td + tf
        values.sort()
        values = list(np.diff(values))[1::2]
        Output[lk] = values[:]
    return Output
#------------------------------------------
#Computation of the node strengths
#  lks_data (link_timeline()): object to analyse
def strengths(lks_data):
    w = weights(lks_data)
    list_lk = [lk.display() for lk in lks_data.links()]
    n1,n2 = zip(*list_lk)
    list_n = set(n1+n2)
    Output = {n:0 for n in list_n}
    for lk in lks_data.links():
        Output[lk.i] += w[lk]
        Output[lk.j] += w[lk]
    return Output
#------------------------------------------
#Computation of the node timelines
#  tij_data (tij()): object to analyse
#  dt (int): length of a time step
def node_timelines(tij_data,dt):
    data = tij_data.out()
    list_lk = [(e.link.i,e.link.j) for e in data]
    n1,n2 = zip(*list_lk)
    list_n = set(n1+n2)
    tset = {n:[] for n in list_n}
    for e in data:
        tset[e.link.i].append(e.time) 
        tset[e.link.j].append(e.time) 
    for n in list_n:
        ts = tset[n][:]
        tset[n] = []
        delta = np.diff(ts)
        tau = 1
        u = ts[0]
        for k,d in enumerate(delta):
            if d > dt:
                tset[n].append((u,tau))
                u = ts[k+1]
                tau = 1
            else:
                tau += 1
        tset[n].append((u,tau))
    return tset
#------------------------------------------
#Computation of the node activities, as numbers of activity periods
#  tij_data (tij()): object to analyse
#  dt (int): length of a time step
def activities_0(tij_data,dt):
    ntl_data = node_timelines(tij_data,dt)
    Output = {n:len(ntl_data[n]) for n in ntl_data.keys()}
    return Output
#------------------------------------------
#Computation of the node activities, as equivalent to strengths but for number of contacts instead of weights
#  tij_data (tij()): object to analyse
#  dt (int): length of a time step
def activities(lks_data):
    nc = number_of_contacts(lks_data)
    list_lk = [lk.display() for lk in lks_data.links()]
    n1,n2 = zip(*list_lk)
    list_n = set(n1+n2)
    Output = {n:0 for n in list_n}
    for lk in lks_data.links():
        Output[lk.i] += nc[lk]
        Output[lk.j] += nc[lk]
    return Output
#------------------------------------------
#Computation of the activity durations list
#  lks_data (link_timeline()): object to analyse
def activity_durations(tij_data,dt):
    ntl_data = node_timelines(tij_data,dt)
    Output = {n:[a[1] for a in ntl_data[n]] for n in ntl_data.keys()}
    return Output
#------------------------------------------
#Computation of the inactivity durations list
#  lks_data (link_timeline()): object to analyse
def inactivity_durations(tij_data,dt):
    ntl_data = node_timelines(tij_data,dt)
    Output = {}
    for n in ntl_data.keys():
        values = [a[0] for a in ntl_data[n]]
        values.sort()
        values = list(np.diff(values))
        Output[n] = values[:]
    return Output
#------------------------------------------
#==========================================
#==========================================
#------------------------------------------
#General functions
#------------------------------------------
#General analysis of a dataset
#  lks_data (link_timeline()): object to analyse
#  dt (int): length of a time step
#  save (boolean): if True, the figure is saved using the filename name; otherwise it is only showed
#  filename (string): name of the figure file
def analysis(lks_data,dt,save=True,filename="analysis.pdf"):
    #conversions
    seq_data = link_timeline_to_snapshot_sequence(lks_data,dt)
    tij_data = snapshot_sequence_to_tij(seq_data)
    list_lk = lks_data.links()
    #plot measures
    a = 3
    nX = 6
    nY = 2
    fig = plt.figure(figsize=(nX*a,nY*a))
    r = float(nX)/nY
    xo = 0.03
    xf = 0.03
    yo = r*xo
    yf = r*xf
    mx = 0.03
    my = r*mx
    w = (1. - xo - xf - (nX-1)*mx)/nX
    h = r*w
    #data analysis & plot
    #--activity timeline
    x = xo
    y = yo
    w_tl = 3*w + 2*mx
    ax = fig.add_axes([x,y,w_tl,h])
    data = activity_timeline(seq_data)
    ax.plot(data,'k-')
    xmax = len(data)
    ax.set_xlim((0,xmax))
    ax.set_xlabel("time")
    ax.set_ylabel("number of events")
    #--activity
    x += w_tl + mx
    ax = fig.add_axes([x,y,w,h])
    data = activities(lks_data).values()
    bins = 2.**(np.arange(0,np.log2(max(data)),0.5))
    dist = np.histogram(data,bins=bins,density=True)
    ax.loglog(bins[:-1],dist[0],'ko',mew=1.5,mfc='None')
    ax.set_xlabel("node activity")
    #--activity durations
    x += w + mx
    ax = fig.add_axes([x,y,w,h])
    data = list(it.chain(*activity_durations(tij_data,dt).values()))
    bins = 2.**(np.arange(0,np.log2(max(data)),0.5))
    dist = np.histogram(data,bins=bins,density=True)
    ax.loglog(bins[:-1],dist[0],'ko',mew=1.5,mfc='None')
    ax.set_xlabel("activity duration")
    #--inactivity durations
    x += w + mx
    ax = fig.add_axes([x,y,w,h])
    data = list(it.chain(*inactivity_durations(tij_data,dt).values()))
    bins = 2.**(np.arange(0,np.log2(max(data)),0.5))
    dist = np.histogram(data,bins=bins,density=True)
    ax.loglog(bins[:-1],dist[0],'ko',mew=1.5,mfc='None')
    ax.set_xlabel("inactivity duration")
    #--degrees
    x = xo
    y += h + my
    ax = fig.add_axes([x,y,w,h])
    k = degrees(list_lk)
    data = k.values()
    nbins = 20
    dist = np.histogram(data,bins=nbins,density=True)
    ax.plot(dist[1][:-1],dist[0],'ko',mew=1.5,mfc='None')
    ax.set_xlabel("degree")
    #--s/k
    x += w + mx
    ax = fig.add_axes([x,y,w,h])
    s = strengths(lks_data)
    kmax = max(k.values())+1
    dk = int(ceil(kmax/float(nbins)))
    bins = range(0,kmax,dk)
    tab = [[] for z in bins]
    for n in k.keys():
        tab[int(k[n]/dk)].append(float(s[n])/float(k[n]))
    moy,std = [],[]
    for b in tab:
        if b != []:
            moy.append(np.mean(b))
            std.append(np.std(b))
        else:
            moy.append(0)
            std.append(0)
    ax.errorbar(bins,moy,yerr=std,color='k',marker='o',markeredgewidth=1.5,markerfacecolor='None')
    ax.set_xlabel("$k$")
    ax.set_ylabel("$s/k$")
    #--weights
    x += w + mx
    ax = fig.add_axes([x,y,w,h])
    data = weights(lks_data).values()
    bins = 2.**(np.arange(0,np.log2(max(data)),0.5))
    dist = np.histogram(data,bins=bins,density=True)
    ax.loglog(bins[:-1],dist[0],'ko',mew=1.5,mfc='None')
    ax.set_xlabel("link weight")
    #--number fo contacts
    x += w + mx
    ax = fig.add_axes([x,y,w,h])
    data = number_of_contacts(lks_data).values()
    bins = 2.**(np.arange(0,np.log2(max(data)),0.5))
    dist = np.histogram(data,bins=bins,density=True)
    ax.loglog(bins[:-1],dist[0],'ko',mew=1.5,mfc='None')
    ax.set_xlabel("number of contacts")
    #--contact durations
    x += w + mx
    ax = fig.add_axes([x,y,w,h])
    data = list(it.chain(*contact_durations(lks_data).values()))
    bins = 2.**(np.arange(0,np.log2(max(data)),0.5))
    dist = np.histogram(data,bins=bins,density=True)
    ax.loglog(bins[:-1],dist[0],'ko',mew=1.5,mfc='None')
    ax.set_xlabel("contact duration")
    #--intercontact durations
    x += w + mx
    ax = fig.add_axes([x,y,w,h])
    data = list(it.chain(*intercontact_durations(lks_data).values()))
    bins = 2.**(np.arange(0,np.log2(max(data)),0.5))
    dist = np.histogram(data,bins=bins,density=True)
    ax.loglog(bins[:-1],dist[0],'ko',mew=1.5,mfc='None')
    ax.set_xlabel("intercontact duration")
    #save/show
    if save:
        plt.savefig(filename)
        plt.close(fig)
    else:
        plt.show()
#------------------------------------------
#==========================================
#==========================================
#------------------------------------------
