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
#from graph_tool.draw import sfdp_layout,graph_draw
import numpy as np
import networkx as nx
#import graph_tool as gt
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
#-filename (string): path+filename
#>returns a tij() object
def read_tij(filename):
    Data = np.loadtxt(filename,delimiter="\t",dtype="int")
    Output = tij()
    for t,i,j in Data:
        Output.add_event(t,i,j)
    return Output
#------------------------------------------
#Reading tijtau.dat
#-filename (string): path+filename
#>returns a tijtau() object
def read_tijtau(filename):
    Data = np.loadtxt(filename,delimiter="\t",dtype="int")
    Output = tijtau()
    for t,i,j,tau in Data:
        Output.add_contact(t,i,j,tau)
    return Output
#------------------------------------------
#Reading snapshot_sequence.dat
#-filename (string): path+filename
#-t_i (int):
#>returns a snapshot_sequence() object
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
#-filename (string): path+filename
#>returns a link_timeline() object
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
#-filename (string): path+filename
#-tij_data (tij()): object to write
def write_tij(filename,tij_data):
    output = open(filename,'w')
    data = tij_data.display()
    for t,i,j in data:
        output.write(str(t)+"\t"+str(i)+"\t"+str(j)+"\n")
    output.close()
#------------------------------------------
#Writing tijtau.dat
#-filename (string): path+filename
#-tijtau_data (tijtau()): object to write
def write_tijtau(filename,tijtau_data):
    output = open(filename,'w')
    data = tijtau_data.display()
    for t,i,j,tau in data:
        output.write(str(t)+"\t"+str(i)+"\t"+str(j)+"\t"+str(tau)+"\n")
    output.close()
#------------------------------------------
#Writing snapshot_sequence.dat
#-filename (string): path+filename
#-seq_data (snapshot_sequence()): object to write
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
#-filename (string): path+filename
#-lks_data (lks_data()): object to write
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
#-tij_data (tij()): object to convert
#-dt (int): time step
#-t_i (int): starting time (optional, default: first time of the file)
#-t_f (int): ending time (optional, default: last time of the file)
#>returns a snapshot_sequence() object
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
#-seq_data (snapshot_sequence()): object to convert
#>returns a tij() object
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
#-tijtau_data (tijtau()): object to convert
#>returns a link_timeline() object
def tijtau_to_link_timeline(tijtau_data):
    Output = link_timeline(list(set([(e.link.i,e.link.j) for e,tau in tijtau_data.out()])))
    for e,tau in tijtau_data.out():
        Output.add_contact(e.link.i,e.link.j,e.time,tau)
    return Output
#------------------------------------------
#Conversion link_timeline->tijtau
#-lks_data (link_timeline()): object to convert
#>returns a tijtau() object
def link_timeline_to_tijtau(lks_data):
    Output = tijtau()
    for lk in lks_data.links():
        for c in lks_data.data[lk]:
            Output.add_contact(c.time,lk.i,lk.j,c.duration)
    return Output
#------------------------------------------
#Conversion tij->tijtau
#-tij_data (tij()): object to convert
#-dt (int): time step
#-join (boolean): indicates whether consecutive instant-events should be joined or not (default: True)
#>returns a tijtau() object
def tij_to_tijtau(tij_data,dt,join=True):
    list_lk = set([e.link for e in tij_data.out()])
    tset = {lk:[] for lk in list_lk}
    for e in tij_data.out():
        tset[e.link].append(e.time)
    Output = tijtau()
    if join:
        for lk in list_lk:
            ts = tset[lk]
            delta = np.diff(ts)
            tau = dt
            u = ts[0]
            for k,d in enumerate(delta):
                if d > dt:
                    Output.add_contact(u,lk.i,lk.j,tau)
                    u = ts[k+1]
                    tau = dt
                else:
                    tau += dt
            Output.add_contact(u,lk.i,lk.j,tau)
    else:
        for lk in list_lk:
            for t in tset[lk]:
                Output.add_contact(t,lk.i,lk.j,dt)
    return Output
#------------------------------------------
#Conversion tijtau->tij
#-tijtau_data (tijtau()): object to convert
#-dt (int): time step
#>returns a tij() object
def tijtau_to_tij(tijtau_data,dt):
    Output = tij()
    for e,tau in tijtau_data.out():
        for t in range(e.time,e.time + tau,dt):
            Output.add_event(t,e.link.i,e.link.j)
    return Output
#------------------------------------------
#Conversion snapshot_sequence->link_timeline
#-seq_data (snapshot_sequence()): object to convert
#-dt (int): time step
#-join (boolean): indicates whether consecutive instant-events should be joined or not (default: True)
#>returns a link_timeline() object
def snapshot_sequence_to_link_timeline(seq_data,dt,join=True):
    list_lk = set().union(*[s[1] for s in seq_data.out()])
    tset = {lk:[] for lk in list_lk}
    for s in seq_data.out():
        t = s[0]
        for lk in s[1]:
            tset[lk].append(t)
    Output = link_timeline([lk.display() for lk in list_lk])
    if join:
        for lk in list_lk:
            ts = tset[lk]
            delta = np.diff(ts)
            tau = dt
            u = ts[0]
            for k,d in enumerate(delta):
                if d > dt:
                    Output.add_contact(lk.i,lk.j,u,tau)
                    u = ts[k+1]
                    tau = dt
                else:
                    tau += dt
            Output.add_contact(lk.i,lk.j,u,tau)
    else:
        for lk in list_lk:
            for t in tset[lk]:
                Output.add_contact(lk.i,lk.j,t,dt)
    return Output
#------------------------------------------
#Conversion link_timeline->snapshot_sequence
#-lks_data (link_timeline()): object to convert
#-dt (int): length of a time step
#-t_i (int): initial time step (optional, default first time step of the file)
#-t_f (int): final time step (optional, default last time step of the file)
#>returns a snapshot_sequence() object
def link_timeline_to_snapshot_sequence(lks_data,dt,t_i=-1,t_f=0):
    data = lks_data.out()
    if t_i < 0:
        t_i = min([lk[1][0].time for lk in data])
    if t_f == 0:
        t_f = max([lk[1][-1].time + lk[1][-1].duration for lk in data]) + dt
    Output = snapshot_sequence(t_i,t_f,dt)
    for lk in lks_data.links():
        for c in lks_data.data[lk]:
            for t in range(c.time,c.time + c.duration,dt):
                Output.data[t].add_link(lk.i,lk.j)
    return Output
#------------------------------------------
#------------------------------------------
#Aggregation tij
#-tij_data (tij()): object to aggregate
#>returns a networkx Grapĥ() object
def aggregate_tij(tij_data):
    G = nx.Graph()
    for e in tij_data.data:
        n,p = e.link.i,e.link.j
        if G.has_edge(n,p):
            G[n][p]['w'] += 1.
        else:
            G.add_edge(n,p,w = 1.)
    return G
#------------------------------------------
#Aggregation tijtau
#- tijtau_data (tijtau()): object to aggregate
#>returns a networkx Grapĥ() object
def aggregate_tijtau(tijtau_data):
    G = nx.Graph()
    for e in tijtau_data.data:
        n,p = e.link.i,e.link.j
        tau = float(tijtau_data.data[e])
        if G.has_edge(n,p):
            G[n][p]['w'] += tau
        else:
            G.add_edge(n,p,w = float(tau))
    return G
#------------------------------------------
#Aggregation snapshot sequence
#-seq_data (snapshot_sequence()): object to aggregate
#>returns a networkx Grapĥ() object
def aggregate_snapshot_sequence(seq_data):
    G = nx.Graph()
    for snapshot in list(seq_data.data.values()):
        for link in snapshot.list_link:
            n,p = link.i,link.j
            if G.has_edge(n,p):
                G[n][p]['w'] += 1.
            else:
                G.add_edge(n,p,w = 1.)
    return G
#------------------------------------------
#Aggregation link timeline
#-lks_data (link_timeline()): object to aggregate
#>returns a networkx Grapĥ() object
def aggregate_link_timeline(lks_data):
    G = nx.Graph()
    for link in lks_data.links():
        w = sum([c.duration for c in lks_data.data[link]])
        G.add_edge(link.i,link.j,w = w)
    return G
#------------------------------------------
#==========================================
#==========================================
#------------------------------------------
#Utilities: Plot
#------------------------------------------
#Plot of a graph
#-G: networkx Graph()
#-node_color: dictionary {node: int}
#-node_shape: dictionary {node: int}
#-edge_width: dictionary {(node,node): float}
#-ax: matplotlib Axes() instance
#-name: string, to name the output files
#-save: boolean, to indicate whether to save the graph as XML or not

"""
def plot_graph(G,node_color={},node_shape={},edge_width={},ax=None,name="graph",save=False):
    nodes = G.nodes()
    nN = len(nodes)
    index = {nodes[i]:i for i in range(nN)}
    #graph for plotting
    G0 = gt.Graph(directed=False)
    v_id = G0.new_vertex_property("int") #node ID
    v_co = G0.new_vertex_property("int") #node color
    if node_color == {}:
        color = {n:0 for n in nodes}
    else:
        color = node_color
    v_sh = G0.new_vertex_property("int") #node shape
    if node_shape == {}:
        shape = {n:0 for n in nodes}
    else:
        shape = node_shape
    vlist = []
    e_w = G0.new_edge_property("float") #edge weight
    if edge_width == {}:
        width = {e:1 for e in G.edges()}
    else:
        width = edge_width
    for n in nodes:
        v = G0.add_vertex()
        v_id[v] = n
        v_co[v] = color[n]
        v_sh[v] = shape[n]
        vlist.append(v)
    for n,p in G.edges():
        i,j = index[n],index[p]
        e = G0.add_edge(vlist[i],vlist[j])
        e_w[e] = width[(n,p)]
#    G0.vertex_properties["ID"] = v_id
#    G0.vertex_properties["Shape"] = v_ta
#    G0.vertex_properties["Color"] = v_gp
#    G0.edge_properties["Weight"] = e_w
    if save:
        G0.save(name+".xml.gz")
    #plot graph
    pos = sfdp_layout(G0,eweight=e_w)
    if ax == None:
        graph_draw(G0,pos,output_size=(1000,1000),
                   vertex_fill_color=v_co,
                   vertex_shape=v_sh,
                   vertex_size=15,
                   edge_pen_width=e_w,
                   bg_color=[1., 1., 1., 1.],
                   output=name+".png"
        )
    else:
        graph_draw(G0,pos,output_size=(1000,1000),
                   vertex_fill_color=v_co,
                   vertex_shape=v_sh,
                   vertex_size=15,
                   edge_pen_width=e_w,
                   bg_color=[1., 1., 1., 1.],
                   mplfig=ax
        )
#------------------------------------------
"""
#==========================================
#==========================================
#------------------------------------------
