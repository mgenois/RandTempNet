#!/usr/bin/env python
# -*- coding: utf-8 -*-
#------------------------------------------
#-     Random Temporal Networks v0.3      -
#-           by Mathieu GÉNOIS            -
#-       genois.mathieu@gmail.com         -
#------------------------------------------
#Python module of randomization techniques for temporal networks
#ref: Gauvin, Génois, Karsai, Kivelä, Takaguchi, Valdano and Vestergaard
#------------------------------------------
#==========================================
#==========================================
#------------------------------------------
#Libraries
import numpy as np
import itertools as it
import matplotlib.pyplot as plt
from math import ceil
from random import choice,sample,randint
from copy import deepcopy
#------------------------------------------
#==========================================
#==========================================
#------------------------------------------
#Class definitions
#------------------------------------------
#link(): tuple of two nodes
#    link.i: int
#    link.j: int
#-Initialisation: link(i,j)
#-Methods:
#    link.display() returns (i,j), with i < j
#-Other:
#    A link() instance is hashable.
#    Comparisons between link() instances follow the rules for undirected links: (u,v) = (v,u).
class link:
    def __init__(self,i,j):
        self.i = min(i,j)
        self.j = max(i,j)
    def __eq__(self,other):
        return self.display() == other.display()
    def __ne__(self,other):
        return self.display() != other.display()
    def __hash__(self):
        return hash(self.display())
    def display(self):
        return (self.i,self.j)
#------------------------------------------
#event(): link i,j active at a time t
#    event.time: int
#    event.link: link()
#-Initialisation: event(t,i,j)
#-Methods:
#    event.change_link(i,j) updates the link with the new i,j nodes
#    event.out() returns (t,link(i,j))
#    event.display() returns (t,i,j)
class event:
    def __init__(self,t,i,j):
        self.link = link(i,j)
        self.time = t
    def __eq__(self,other):
        return self.display() == other.display()
    def __ne__(self,other):
        return self.display() != other.display()
    def __hash__(self):
        return hash(self.display())
    def change_link(self,i,j):
        self.link = link(i,j)
    def out(self):
        return (self.time,self.link)
    def display(self):
        return (self.time,self.link.i,self.link.j)
#------------------------------------------
#contact(): activation at a time t with a duration tau
#    contact.time: int
#    contact.duration: int
#-Initialisation: contact(t,tau)
#-Methods:
#    contact.out() returns (t,tau)
#    contact.display() returns (t,tau)
class contact:
    def __init__(self,t,tau):
        self.time = t
        self.duration = tau
    def __eq__(self,other):
        return self.display() == other.display()
    def __ne__(self,other):
        return self.display() != other.display()
    def __hash__(self):
        return hash(self.display())
    def display(self):
        return (self.time,self.duration)
#------------------------------------------
#-snapshot(): set of links.
#    snapshot.list_link: set of link()
#-Initialisation: snapshot(t,list_link)
#-Methods:
#    snapshot.add_link(i,j) adds a link(i,j) to list_link
#    snapshot.del_link(i,j) removes the link(i,j) from list_link
#    snapshot.add_links(list_lk) adds all links in list_lk to list_link
#    snapshot.del_links(list_lk) removes all links in list_lk from list_link
#    snapshot.out() returns set([link()...])
#    snapshot.display() returns [(i,j)...]
#-Other:
#    As list_link is a set, the unicity of links is always preserved.
class snapshot:
    def __init__(self,list_link):
        self.list_link = set(list_link)
    def add_link(self,i,j):
        self.list_link |= set([link(i,j)])
    def del_link(self,i,j):
        self.list_link -= set([link(i,j)])
    def add_links(self,list_lk):
        values = [link(i,j) for i,j in list_lk]
        self.list_link |= set(values)
    def del_links(self,list_lk):
        values = [link(i,j) for i,j in list_lk]
        self.list_link -= set(values)
    def out(self):
        return self.list_link
    def display(self):
        return [(lk.i,lk.j) for lk in self.list_link]
#------------------------------------------
#tij(): list of events ordered in time.
#    tij.data: list of events
#-Initialisation: tij()
#-Methods:
#    tij.add_event(t,i,j) adds an event (t,i,j)
#    tij.del_event(t,i,j) removes the event (t,i,j)
#    tij.add_events(list_e) adds an all events in list_e
#    tij.del_events(list_e) removes all events in list_e
#    tij.out() returns [event(),...] sorted by time
#    tij.display() returns [(t,i,j),...] sorted by time
class tij:
    def __init__(self):
        self.data = set([])
    def add_event(self,t,i,j):
        self.data |= set([event(t,i,j)])
    def del_event(self,t,i,j):
        self.data -= set([event(t,i,j)])
    def add_events(self,list_e):
        values = [event(t,i,j) for t,i,j in list_e]
        self.data |= set(values)
    def del_events(self,list_e):
        values = [event(t,i,j) for t,i,j in list_e]
        self.data -= set(values)
    def out(self):
        values = sorted(list(self.data),key=lambda x:x.time)
        return values
    def display(self):
        values = self.out()
        return [e.display() for e in values]
#------------------------------------------
#tijtau(): dictionary of durations tau labeled by triplets (t,i,j)
#    tijtau.data: list of contacts
#-Initialisation: tijtau()
#-Methods:
#    tijtau.add_contact(t,i,j,tau) adds a contact (t,i,j,tau)
#    tijtau.del_contact(t,i,j) removes the contact identified by (t,i,j)
#    tijtau.out() returns [contact(),...] sorted by time
#    tijtau.display() returns [(t,i,j,tau),...]
class tijtau:
    def __init__(self):
        self.data = {}
    def add_contact(self,t,i,j,tau):
        self.data[event(t,i,j)] = tau
    def del_contact(self,t,i,j): 
        del self.data[event(t,i,j)]
    def add_contacts(self,list_c):
        for t,i,j,tau in list_c:
            self.data[event(t,i,j)] = tau
    def del_contacts(self,list_c):
        for t,i,j in list_c:
            del self.data[event(t,i,j)]
    def out(self):
        values = [(e,self.data[e]) for e in self.data.keys()]
        values = sorted(values,key=lambda x:x[0].time)
        return values
    def display(self):
        values = self.out()
        return [(e.time,e.link.i,e.link.j,tau) for e,tau in values]
#------------------------------------------
#snapshot_sequence(): dictionary of snapshots ordered in time
#    snapshot_sequence.data: dictionary of snapshots with time as keys
#-Initialisation: snapshot_sequence(t_i,t_f,dt)
#-Methods:
#    snapshot_sequence.update_snapshot(t,list_link) updates the snapshot at time t with the newlist_link
#    snapshot_sequence.clear_snapshot(t) removes all the links from the snapshot at time t
#    snapshot_sequence.out() returns a list of tuples (t,list_link)
#    snapshot_sequence.display() returns [(t_i,[(i,j)...]...)
class snapshot_sequence:
    def __init__(self,t_i,t_f,dt):
        self.data = {t:snapshot([]) for t in range(t_i,t_f,dt)}
    def update_snapshot(self,t,list_link):
        self.data[t] = snapshot(list_link)
    def clear_snapshot(self,t):
        self.data[t] = snapshot([])
    def out(self):
        list_t = self.data.keys()
        list_t.sort()
        return [(t,list(self.data[t].out())) for t in list_t]
    def display(self):
        list_t = self.data.keys()
        list_t.sort()
        return [(t,[lk.display() for lk in self.data[t].out()]) for t in list_t]
#------------------------------------------
#link_timeline(): dictionary of links with their associated timeline of contacts
#    link_timeline.data: dictionary of links with their associated timeline of activity
#-Initialisation: link_timeline(<list_lk>,<list_tl>) <optional>
#-Methods:
#    link_timeline.links() returns the list of links
#    link_timeline.add_link(i,j,tl) adds a link with its timeline
#    link_timeline.del_link(i,j) removes a link
#    link_timeline.add_links(list_lk,list_tl) adds a list of links with their associated timelines
#    link_timeline.del_links(list_lk) removes a list of links
#    link_timeline.add_contact(i,j,t,tau) adds a contact to the timeline of a link
#    link_timeline.del_contact(i,j,t,tau) removes a contact from the timeline of link
#    link_timeline.out() returns a list of tuples (link(),[contact(),...])
#    link_timeline.display() returns a list of tuples ((i,j),[t...])
class link_timeline:
    def __init__(self,list_lk=[],list_tl=[]):
        if list_tl != []:
            values = zip(list_lk,list_tl)
            self.data = {link(lk[0],lk[1]):set([contact(t,tau) for t,tau in tl]) for lk,tl in values}
        else:
            self.data = {link(i,j):set([]) for i,j in list_lk}
    def links(self):
        return self.data.keys()
    def links_display(self):
        values = self.links()
        return [lk.display() for lk in values]
    def add_link(self,i,j,timeline=[]):
        self.data[link(i,j)] = set([contact(t,tau) for t,tau in timeline])
    def del_link(self,i,j):
        del self.data[link(i,j)]
    def add_links(self,list_lk,list_tl=[]):
        if list_tl != []:
            values = zip(list_lk,list_tl)
            self.data = {link(lk[0],lk[1]):set([contact(t,tau) for t,tau in tl]) for lk,tl in values}
        else:
            self.data = {link(i,j):set([]) for i,j in list_lk}
    def del_links(self,list_lk):
        for i,j in list_lk:
            del self.data[link(i,j)]
    def add_contact(self,i,j,t,tau):
        self.data[link(i,j)] |= set([contact(t,tau)])
    def del_contact(self,i,j,t,tau):
        self.data[link(i,j)] -= set([contact(t,tau)])
    def out(self):
        return [(lk,sorted(self.data[lk],key=lambda x:x.time)) for lk in self.links()]
    def display(self):
        values = self.out()
        return [(lk.display(),[c.display() for c in tl]) for lk,tl in values]
#------------------------------------------
#==========================================
#==========================================
#------------------------------------------
#Data reading & writing
#------------------------------------------
#Formats:
#--tij.dat:               "t \t i \t j \n"
#--tijtau.dat:            "t \t i \t j \t tau \n"
#--snapshot_sequence.dat: "t \t i,j \t k,l \t ... \t y,z \n"
#--link_timeline.dat:       "i,j \t t_1,tau_1 \t t_2,tau_2 \t ... \t t_n,tau_n \n"
#------------------------------------------
#Reading tij.dat
#  filename (string): path+filename
def read_tij(filename):
    Data = np.loadtxt(filename,delimiter="\t",dtype="int")
    Output = tij()
    for t,i,j in Data:
        Output.add_event(t,i,j)
    return Output
#------------------------------------------
#Reading tijtau.dat
#  filename (string): path+filename
def read_tijtau(filename):
    Data = np.loadtxt(filename,delimiter="\t",dtype="int")
    Output = tijtau()
    for t,i,j,tau in Data:
        Output.add_contact(t,i,j,tau)
    return Output
#------------------------------------------
#Reading snapshot_sequence.dat
#  filename (string): path+filename
#  t_i (int): 
def read_snapshot_sequence(filename,t_i=0,t_f=0,dt=0):
    Input = open(filename,'r')
    data = Input.readlines()
    Input.close()
    if dt == 0:
        t_i = int(data[0].split("\t")[0])
        t2 = int(data[1].split("\t")[0])
        t_f = int(data[1].split("\t")[0])
        dt = t2 - t_i
    Output = snapshot_sequence(t_i,t_f,dt)
    for l in data:
        line = l.split("\t")
        t = int(line[0])
        list_link = []
        for lk in line[1:]:
            i,j = lk.split(",")
            list_link.append(link(int(i),int(j)))
        Output.update_snapshot(t,list_link)
    return Output
#------------------------------------------
#Reading link_timeline.dat
#  filename (string): path+filename
def read_link_timeline(filename):
    Input = open(filename,'r')
    Output = link_timeline()
    for l in Input:
        line = l.split("\t")
        lk = line[0].split(",")
        i,j = int(lk[0]),int(lk[1])
        timeline = []
        for c in line[1:]:
            t,tau = c.split(",")
            timeline.append((int(t),int(tau)))
        Output.add_link(i,j,timeline)
    Input.close()
    return Output
#------------------------------------------
#Writing tij.dat
#  filename (string): path+filename
#  tij_data (tij()): object to write
def write_tij(filename,tij_data):
    output = open(filename,'w')
    data = tij_data.display()
    for t,i,j in data:
        output.write(str(t)+"\t"+str(i)+"\t"+str(j)+"\n")
    output.close()
#------------------------------------------
#Writing tijtau.dat
#  filename (string): path+filename
#  tijtau_data (tijtau()): object to write
def write_tijtau(filename,tijtau_data):
    output = open(filename,'w')
    data = tijtau_data.display()
    for t,i,j,tau in data:
        output.write(str(t)+"\t"+str(i)+"\t"+str(j)+"\t"+str(tau)+"\n")
    output.close()
#------------------------------------------
#Writing snapshot_sequence.dat
#  filename (string): path+filename
#  seq_data (snapshot_sequence()): object to write
def write_snapshot_sequence(filename,seq_data):
    output = open(filename,'w')
    for step in seq_data.out():
        t,list_link = step
        output.write(str(t))
        for lk in list_link:
            output.write("\t"+str(lk.i)+","+str(lk.j))
        output.write("\n")
    output.close()
#------------------------------------------
#Writing link_timeline.dat
#  filename (string): path+filename
#  lks_data (lks_data()): object to write
def write_link_timeline(filename,lks_data):
    output = open(filename,'w')
    for lkt in lks_data.out():
        lk,timeline = lkt
        output.write(str(lk.i)+","+str(lk.j))
        for c in timeline:
            output.write("\t"+str(c.time)+","+str(c.duration))
        output.write("\n")
    output.close()
#------------------------------------------
#==========================================
#==========================================
#------------------------------------------
#Data conversion
#------------------------------------------
#Conversion tij->snapshot_sequence
#  tij_data (tij()): object to convert
#  dt (int): time step
#  t_i (int): starting time (optional, default: first time of the file)
#  t_f (int): ending time (optional, default: last time of the file)
def tij_to_snapshot_sequence(tij_data,dt,t_i=-1,t_f=0):
    data = tij_data.out()
    if t_i < 0:
        t_i = data[0].time
    if t_f == 0:
        t_f = data[-1].time + 1
    Output = snapshot_sequence(t_i,t_f,dt)
    for e in data:
        i = e.link.i
        j = e.link.j
        t = e.time
        Output.data[t].add_link(i,j)
    return Output
#------------------------------------------
#Conversion snapshot_sequence->tij
#  seq_data (snapshot_sequence()): object to convert
def snapshot_sequence_to_tij(seq_data):
    Output = tij()
    for step in seq_data.out():
        t,list_link = step
        if list_link != []:
            for lk in list_link:
                Output.add_event(t,lk.i,lk.j)
    return Output
#------------------------------------------
#Conversion tijtau->link_timeline
#  tijtau_data (tijtau()): object to convert
def tijtau_to_link_timeline(tijtau_data):
    Output = link_timeline(list(set([(e.link.i,e.link.j) for e,tau in tijtau_data.out()])))
    for e,tau in tijtau_data.out():
        Output.add_contact(e.link.i,e.link.j,e.time,tau)
    return Output
#------------------------------------------
#Conversion link_timeline->tijtau
#  lks_data (link_timeline()): object to convert
def link_timeline_to_tijtau(lks_data):
    Output = tijtau()
    for lk in lks_data.links():
        for c in lks_data.data[lk]:
            Output.add_contact(c.time,lk.i,lk.j,c.duration)
    return Output
#------------------------------------------
#Conversion tij->tijtau
#  tij_data (tij()): object to convert
#  dt (int): time step
def tij_to_tijtau(tij_data,dt):
    list_lk = set([e.link for e in tij_data.out()])
    tset = {lk:[] for lk in list_lk}
    for e in tij_data.out():
        tset[e.link].append(e.time)
    Output = tijtau()
    for lk in list_lk:
        ts = tset[lk]
        delta = np.diff(ts)
        tau = 1
        u = ts[0]
        for k,d in enumerate(delta):
            if d > dt:
                Output.add_contact(u,lk.i,lk.j,tau)
                u = ts[k+1]
                tau = 1
            else:
                tau += 1
        Output.add_contact(u,lk.i,lk.j,tau)
    return Output
#------------------------------------------
#Conversion tijtau->tij
#  tijtau_data (tijtau()): object to convert
#  dt (int): time step
def tijtau_to_tij(tijtau_data,dt):
    Output = tij()
    for e,tau in tijtau_data.out():
        for t in range(e.time,e.time + tau,dt):
            Output.add_event(t,e.link.i,e.link.j)
    return Output
#------------------------------------------
#Conversion snapshot_sequence->link_timeline
#  seq_data (snapshot_sequence()): object to convert
#  dt (int): time step
def snapshot_sequence_to_link_timeline(seq_data,dt):
    list_lk = set().union(*[s[1] for s in seq_data.out()])
    tset = {lk:[] for lk in list_lk}
    for s in seq_data.out():
        t = s[0]
        for lk in s[1]:
            tset[lk].append(t)
    Output = link_timeline([lk.display() for lk in list_lk])
    for lk in list_lk:
        ts = tset[lk]
        delta = np.diff(ts)
        tau = 1
        u = ts[0]
        for k,d in enumerate(delta):
            if d > dt:
                Output.add_contact(lk.i,lk.j,u,tau)
                u = ts[k+1]
                tau = 1
            else:
                tau += 1
        Output.add_contact(lk.i,lk.j,u,tau)
    return Output
#------------------------------------------
#Conversion link_timeline->snapshot_sequence
#  lks_data (link_timeline()): object to convert
#  dt (int): length of a time step
#  t_i (int): initial time step (optional, default first time step of the file)
#  t_f (int): final time step (optional, default last time step of the file)
def link_timeline_to_snapshot_sequence(lks_data,dt,t_i=-1,t_f=0):
    data = lks_data.out()
    if t_i < 0:
        t_i = min([lk[1][0].time for lk in data])
    if t_f == 0:
        t_f = max([lk[1][-1].time + lk[1][-1].duration for lk in data]) + 1
    Output = snapshot_sequence(t_i,t_f,dt)
    for lk in lks_data.links():
        for c in lks_data.data[lk]:
            for t in range(c.time,c.time + c.duration,dt):
                Output.data[t].add_link(lk.i,lk.j)
    return Output
#------------------------------------------
#==========================================
#==========================================
#------------------------------------------
#Utilities
#------------------------------------------
#Computation of the degrees of the nodes
#  list_lk: list of link() objects
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
#  list_lk: list of link() objects
#  group: dictionary of group affiliations
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
#Computation of the number of contacts for each link
#  lks_data (link_timeline()): object to analyse
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
#Computation of the node timelines
#  tij_data (tij()): object to analyse
#  dt (int): length of a time step
def activities(tij_data,dt):
    ntl_data = node_timelines(tij_data,dt)
    Output = {n:len(ntl_data[n]) for n in ntl_data.keys()}
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
    data = activities(tij_data,dt).values()
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
    else:
        plt.show()
#------------------------------------------
#==========================================
#==========================================
#------------------------------------------
#Randomization functions
#------------------------------------------
#==========================================
#------------------------------------------
#Snapshot sequence representation
#--All the following functions take a snapshot_sequence as input,
#  and return a snapshot_sequence.
#------------------------------------------
#P__pGt_sgnE (Permutation of active snapshots)
#  seq_data (snapshot_sequence())
def P__pGt_sgnE(seq_data):
    #extraction of snapshot_sequence caracteristics
    list_time = seq_data.data.keys()
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
#P__pGt (Permutation of snapshots)
#  seq_data (snapshot_sequence())
def P__pGt(seq_data):
    #extraction of snapshot_sequence caracteristics
    list_time = seq_data.data.keys()
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
    list_t = sample(range(t_i,t_f,dt),len(list_S))
    #reconstruction
    Output = snapshot_sequence(t_i,t_f,dt)
    for t in list_t:
        list_link = list_S.pop()
        Output.update_snapshot(t,list_link)
    return Output
#------------------------------------------
#P__isoGt_Phi (Permutation of active node identities in snapshots)
#  seq_data (snapshot_sequence())
def P__isoGt_Phi(seq_data):
    #construction of the new snapshot_sequence
    list_time = seq_data.data.keys()
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
#P__isoGt (Permutation of node identities in each snapshot)
#  seq_data (snapshot_sequence())
def P__isoGt(seq_data):
    #construction of the new snapshot_sequence
    list_time = seq_data.data.keys()
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
#P__Phi (Permutation of links between active nodes in each snapshot)
#  seq_data (snapshot_sequence())
def P__Phi(seq_data):
    #construction of the new snapshot_sequence
    list_time = seq_data.data.keys()
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
#P__E (Permutation of links in each snapshot)
#  seq_data (snapshot_sequence())
def P__E(seq_data):
    #construction of the new snapshot_sequence
    list_time = seq_data.data.keys()
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
#P__muE (Global permutation of events)
#  seq_data (snapshot_sequence())
def P__muE(seq_data):
    #construction of the new snapshot_sequence
    list_time = seq_data.data.keys()
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
    list_t = range(t_i,t_f,dt)
    #permutation
    for k in range(nE):
        t = choice(list_t)
        i,j = sample(nodes,2)
        while link(i,j) in Output.data[t].out():
            i,j = sample(nodes,2)
        Output.data[t].add_link(i,j)
    return Output
#------------------------------------------
#P__k (Permutation of links with degree preservation in each snapshot)
#  seq_data (snapshot_sequence())
#  link_threshold (int): minimum number of links to use the configuration model algorithm
#  n_iter (int): parameter for the number of interations in the case of the Sneppen-Maslov algorithm
def P__k(seq_data,link_threshold=20,n_iter=5):
    #extraction of snapshots to shuffle
    todo = [s for s in seq_data.out() if len(s[1]) > 1]
    for t,list_link in todo:
        #configuration model
        if len(list_link) >= link_threshold:
            #extraction of degrees and node list
            deg = degrees(list_link)
            nodes = deg.keys()
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
#==========================================
#------------------------------------------
# Link timeline representation
#--All the following functions take a link_timeline as input,
#  and return a link_timeline.
#------------------------------------------
# Link permutations
#------------------------------------------
#P__kstat_pTheta (Permutation of links with degree preservation)
#  lks_data (link_timeline())
def P__kstat_pTheta(lks_data):
    list_links = lks_data.links()
    #extraction of the timelines
    list_timeline = [[c.display() for c in lks_data.data[lk]] for lk in list_links]
    #extraction of degrees and node list
    deg = degrees(list_links)
    nodes = deg.keys()
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
            if len(list_nodes) >= deg[n]:
                neighbors = sample(list_nodes,deg[n])
                deg[n] = 0
                for p in neighbors:
                    deg[p] -= 1
                    Tl = choice(list_tl)
                    list_tl.remove(Tl)
                    Output.add_link(n,p,Tl)
                #updating the list of available nodes
                list_nodes = [n for n in list_nodes if deg[n] > 0]
                list_nodes = sorted(list_nodes,key=lambda x:deg[x])
            else:
                list_nodes = []
                redo = True
    return Output
#------------------------------------------
#P__LCM (Permutation of links with group structure preservation)
#  lks_data (link_timeline()):
#  group: dictionary associating each node with its group. Group labels can be of any type.
def P__LCM(lks_data,group):
    #extraction of the group labels
    group_labels = list(set(group.values()))
    #dictionary of node indices
    nodes = group.keys()
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
#P__kstat_LCM (Permutation of links with group structure and node degrees preservation)
#  lks_data: (link_timeline())
#  group: dictionary associating each node with its group. Group labels can be of any type.
#  n_iter: factor for the number of iterations: number of permutations = n_iter x number of links. By default, n_iter = 5.
def P__kstat_LCM(lks_data,group,n_iter=3):
    list_links = lks_data.links()
    Output = link_timeline(lks_data.links_display())
    nL = len(list_links)
    #initialisation of the convergence tracking
    converge = [1]
    #extraction of the group labels
    group_labels = list(set(group.values()))
    nG = len(group_labels)
    #extraction of the nodes
    nodes = group.keys()
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
# Timeline permutations
#------------------------------------------
#P__n_pTheta (Permutation of timelines with preservation of the contact frequencies)
#  lks_data (link_timeline())
def P__n_pTheta(lks_data):
    #extraction of the contact frequencies
    dict_link = number_of_contacts(lks_data)
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
#P__w_pTheta (Permutation of timelines with preservation of the weights)
#lks_data (link_timeline())
def P__w_pTheta(lks_data):
    #extraction of the contact frequencies
    dict_link = weights(lks_data)
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
#P__Gstat_pTheta (Permutation of timelines)
#lks_data: (link_timeline())
def P__Gstat_pTheta(lks_data):
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
#P__pitau_pidtau_t1 (Interval shuffling in place with conservation of the initial time)
#  lks_data (link_timeline())
def P__pitau_pidtau_t1(lks_data):
    Output = link_timeline()
    dict_tau = contact_durations(lks_data)
    dict_dtau = intercontact_durations(lks_data)
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
#P__pitau_pidtau (Interval shuffling in place)
#  lks_data (link_timeline())
#  ti (int): first time step of the dataset
#  tf (int): last time step of the dataset
#  dt (int): duration of a time step
def P__pitau_pidtau(lks_data,ti,tf,dt):
    Output = link_timeline()
    dict_tau = contact_durations(lks_data)
    dict_dtau = intercontact_durations(lks_data)
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
#P__pitau (Contacts shuffling in place)
#  lks_data (link_timeline())
#  ti (int): first time step of the dataset
#  tf (int): last time step of the dataset
#  dt (int): duration of a time step
def P__pitau(lks_data,ti,tf,dt):
    Output = link_timeline()
    dict_tau = contact_durations(lks_data)
    for lk in lks_data.links():
        #extraction of the contact events
        contacts = dict_tau[lk]
        #computation of the remaining usable time
        tu = tf - ti - sum(contacts)
        #construction of the new timeline
        contacts = sample(contacts,len(contacts))
        #new starting times for the contacts
        list_t = [randint(0,tu/dt)*dt for c in contacts]
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
#P__n_ptau_E (Global contacts shuffling)
#  lks_data (link_timeline())
#  dt (int): duration of a time step
def P__n_ptau_E(lks_data,dt):
    #determination of the number of contacts
    num = number_of_contacts(lks_data)
    #link ordering by number of contact
    list_lk = sorted(lks_data.links(),key=lambda x:num[x],reverse=True)
    #contacts extraction
    list_c = list(it.chain(*lks_data.data.values()))
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
        loc_c = range(ti-dt,tf+dt,dt) #extended list of activation times
        list_c.remove(c)
        Tl.append(c.display())
        testlist_t = loc_c[:] #test list for overlapping
        for ic in range(n-1):
            c = choice(list_c)
            ti = c.time
            tf = c.time + c.duration
            loc_c = range(ti-dt,tf+dt,dt)
            #test for overlapping and concatenation
            test = testlist_t + loc_c
            while len(test) > len(list(set(test))):
                c = choice(list_c)
                ti = c.time
                tf = c.time + c.duration
                loc_c = range(ti-dt,tf+dt,dt)
                test = testlist_t + loc_c
            #adding the contact
            testlist_t += loc_c
            list_c.remove(c)
            Tl.append(c.display())
        #adding the new timeline
        Output.add_link(lk.i,lk.j,Tl)
    return Output
#------------------------------------------
#P__ptau_E (Global contacts shuffling)
#  lks_data (link_timeline())
#  dt: duration of a time step (int)
def P__ptau_E(lks_data,dt):
    #links extraction
    list_lk = list(lks_data.links())
    #contacts extraction with time stamps
    list_c = list(it.chain(*lks_data.data.values()))
    #contacts redistribution
    Output = link_timeline(lks_data.links_display())
    Tl = {lk:[] for lk in list_lk}
    for c in list_c:
        lk = choice(list_lk)
        #virtual extension of the contact to test for concatenation
        ti = c.time
        tf = c.time + c.duration
        loc_c = range(ti-dt,tf+dt,dt) #extended list of activation times
        #test for overlapping
        test = np.array([t in Tl[lk] for t in loc_c])
        while test.any():
            lk = choice(list_lk)
            ti = c.time
            tf = c.time + c.duration
            loc_c = range(ti-dt,tf+dt,dt) #extended list of activation times
            test = np.array([t in Tl[lk] for t in loc_c])
        Output.add_contact(lk.i,lk.j,c.time,c.duration)
        Tl[lk] += loc_c
    return Output
#------------------------------------------
#P__w_E (Global events shuffling)
#  lks_data (link_timeline())
#  dt (int): duration of a time step
def P__w_E(lks_data,dt):
    #link weights extraction
    weight = weights(lks_data)
    #link ordering by weight
    list_lk = sorted(lks_data.links(),key=lambda x:weight[x],reverse=True)
    #events extraction
    list_t = []
    for lk in list_lk:
        for c in lks_data.data[lk]:
            ti = c.time
            tf = c.time + c.duration
            list_t += range(ti,tf,dt)
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
#P__w (Global events permutation with weights preservation)
#  lks_data (link_timeline())
#  ti (int): first time step of the dataset
#  tf (int): last time step of the dataset
#  dt (int): duration of a time step
def P__w(lks_data,ti,tf,dt):
    #events redistribution
    Output = tij()
    #link weights extraction
    weight = weights(lks_data)
    for lk in lks_data.links():
        w = weight[lk]
        list_t = sample(range(ti,tf,dt),w)
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
    weight = weights(lks_data)
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
            list_t = sample(range(t1+1,tw,dt),w-2)
            for t in list_t:
                Output.add_event(t,lk.i,lk.j)
    return Output
#------------------------------------------
#P__Gstat (Global events permutation)
#  lks_data (link_timeline())
#  ti (int): first time step of the dataset
#  tf (int): last time step of the dataset
#  dt (int): duration of a time step
def P__Gstat(lks_data,ti,tf,dt):
    #extraction of the total number of contacts
    nC = sum(weights(lks_data).values())
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
#P__pi_t_tau (Contacts shuffling according to starting time)
#  lks_data (link_timeline())
#  dt (int): duration of a time step
def P__pi_t_tau(lks_data,dt):
    Output = link_timeline()
    dict_tau = contact_durations(lks_data)
    for lk in lks_data.links():
        #extraction of the contact events
        contacts = dict_tau[lk]
        #computation of the remaining usable time
        tu = tf - ti - sum(contacts)
        #construction of the new timeline
        contacts = sample(contacts,len(contacts))
        #new starting times for the contacts
        list_t = [randint(0,tu/dt)*dt for c in contacts]
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
#------------------------------------------
