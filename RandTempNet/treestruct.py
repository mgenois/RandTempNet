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
import networkx as nx
import numpy as np
#import graph_tool as gt
import matplotlib.pyplot as plt
#from graph_tool.draw import sfdp_layout,graph_draw,arf_layout
from .utils import *
#------------------------------------------
#==========================================
#==========================================
#------------------------------------------
#Computing the tree structure of a network by depercolating its links according to increasing weights
#------------------------------------------
#Computation of the depercolation steps according to link weight
#-path: path to the tij file
def deperc_steps_w(Gb):
#    tij_data = read_tij(path)
#    G = aggregate_tij(tij_data)
    G = Gb.copy()
    #sorting of the edges according to their weight
    weight = nx.get_edge_attributes(G,'w')
    lW = list(set(weight.values()))
    lW.sort()
    dE = {w:[] for w in lW}
    for e in G.edges():
        dE[weight[e]].append(e)
    #computation of the depercolation steps
    nC = nx.number_connected_components(G)
    thG,thE,thW = [],[],[]
    for w in lW:
        prevG = G.copy()
        G.remove_edges_from(dE[w])
        n = nx.number_connected_components(G)
        if n > nC:
            nC = n
            thG.append(prevG)
            thE.append(dE[w])
            thW.append(w)
    return thG,thE,thW
#------------------------------------------
#Computation of the depercolation steps according to link centrality
#-path: path to the tij file
def deperc_steps_bc(Gb):
    G = Gb.copy()
    thG,thE,thC = [],[],[]
    nC = nx.number_connected_components(G)
    while G.number_of_edges() > 0:
        #sorting of the edges according to their centrality
        central = nx.edge_betweenness_centrality(G)
#        central = nx.edge_betweenness_centrality(G,weight="w")
        lC = list(central.values())
        cmax = max(lC)
        lE = [e for e in G.edges() if central[e] == cmax]
        #computation of the depercolation steps
        prevG = G.copy()
        G.remove_edges_from(lE)
        n = nx.number_connected_components(G)
        if n > nC:
            nC = n
            thG.append(prevG)
            thE.append(lE)
            thC.append(1./cmax)
    return thG,thE,thC
#------------------------------------------
#Distances within a tree
#-thG: list of networkx Graph() objects
#-thW: list of weights
#-ncol: number of columns for the plot
#-node_color: dictionary {node: int}
#-node_shape: dictionary {node: int}
#-edge_width: dictionary {(node,node): float}
#-size: size of a single graph (int)
def distances(thG,thW,name="distances.png"):
    Tree,depth = make_tree(thG,thW)
    Tree = Tree.to_undirected()
    lN = list(thG[0].nodes())
    lN.sort()
    #finding the leaves
    dN = {}
    for n in Tree.nodes():
        if n > 0:
            if len(Tree.node[n]['e']) == 1:
                dN[Tree.node[n]['e'][0]] = n
    nN = len(lN)
    tab = []
    for i in range(nN):
        lDist = []
        for j in range(nN):
            lDist.append([nx.shortest_path_length(Tree,source=dN[lN[i]],target=dN[lN[j]]),lN[j]])
        lDist.sort()
        l1,l2 = list(zip(*lDist))
        tab.append(l2)
        tab.append(l1)
    np.savetxt(name,tab,fmt = "%d",delimiter = "\t")
#------------------------------------------
#Plot of the graph steps
#-thG: list of networkx Graph() objects
#-thE: list of edge lists
#-ncol: number of columns for the plot
#-node_color: dictionary {node: int}
#-node_shape: dictionary {node: int}
#-edge_width: dictionary {(node,node): float}
#-size: size of a single graph (int)
"""
def plot_graph_steps(thG,thE,name="graph_steps.pdf",ncol=5,node_color={},node_shape={},edge_width={},size=2):
    plt.switch_backend('cairo')
    nG = len(thG)
    nrow = int(nG/ncol)+1
    a = size #size of a single graph
    fig = plt.figure(figsize=(ncol*a,nrow*a))
    w = 1./ncol
    h = 1./nrow
    for z,G in enumerate(thG):
        #position of the graph
        x = (z%ncol)*w
#        y = 1.-h*(1+z/ncol)
        y = (z/ncol)*h
        ax = fig.add_axes([x,y,w,h])
        #plot preparation
        nodes = list(G.nodes())
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
        e_c = G0.new_edge_property("string") #edge color
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
            if (n,p) in thE[z]:
                e_c[e] = '#ff0000'
            else:
                e_c[e] = '#000000'
        #plot graph
        pos = sfdp_layout(G0,eweight=e_w)
        graph_draw(G0,pos,
                   vertex_fill_color=v_co,
                   vertex_shape=v_sh,
                   vertex_size=5,
                   edge_pen_width=1,
                   edge_color=e_c,
                   mplfig=ax
        )
        ax.set_aspect('equal')
        ax.axis('off')
    plt.savefig(name)
#------------------------------------------
"""
def make_tree(thG,thW):
    Tree = nx.DiGraph()
    Tree.add_node(-1,w=0,e=[-1],x=0,corner=[0,0])
    Tree.add_node(0,w=0,e=list(thG[0].nodes()))
    Tree.add_edge(-1,0)
    active_N = [0]
    k = 1
    w0 = max(thW) + 10
    #boucle sur les steps
    for i,G in enumerate(thG[1:]):
        remove_N = []
        #boucle sur les composantes de chaque step
        for subG in nx.connected_components(G):
            #boucle sur les Nodes pendants
            for N in active_N:
                #test d'inclusion des noeuds du composant dans N
                test = np.array([p in Tree.node[N]["e"] for p in subG])
                if test.all():
                    #test d'égalité des sets
                    if len(subG) < len(Tree.node[N]["e"]):
                        #test de leaf
                        if len(subG) == 1:
                            Tree.add_node(k,w=w0,e=list(subG))
                        else:
                            Tree.add_node(k,w=0,e=list(subG))
                            active_N.append(k)
                        Tree.node()[N]["w"] = thW[i]
                        Tree.add_edge(N,k)
                        remove_N.append(N)
                        k+=1
        remove_N = set(remove_N)
        for N in remove_N:
            active_N.remove(N)
    #ajout des leaves
    for n in thG[0].nodes():
        #boucle sur les Nodes pendants
        for N in active_N:
            #test d'appartenance à N
            if n in Tree.node[N]["e"]:
                Tree.add_node(k,w=w0,e=[n])
                Tree.add_edge(N,k)
                Tree.node()[N]["w"] = thW[-1]
                k+=1

    depth = set_depth(Tree)
    
    return Tree,depth
#------------------------------------------
def set_depth(Tree):
    depth = 0
    to_treat = [0]
    while to_treat != []:
        n = to_treat.pop()
        Tree.node()[n]["x"] = depth
        Tree.node()[n]["corner"] = [depth,Tree.node()[n]["w"]]
        suiv = list(Tree.successors(n))
        #cas d'une fin de branche -> increment de depth
        if suiv == []:
            depth += 1
        #sinon -> ajout des nouveaux noeuds a traiter
        else:
            # tri par taille du groupe
            suiv = sorted(suiv,key=lambda x:len(Tree.node[x]["e"]))
            to_treat += suiv
    return depth
#------------------------------------------
def find_ta(thG,thW,n_ta): 
    for i,G in enumerate(thG):
        for subG in nx.connected_components(G):
            if list(subG) == [n_ta]:
                return thW[i-1]
#------------------------------------------
"""
#Plot of the tree structure
def plot_tree(thG,thW,name,n_ta=-1):
    global depth
    plt.switch_backend('cairo')

    Tree,depth = make_tree(thG,thW)
    if n_ta >= 0:
        th_ta = find_ta(thG,thW,n_ta)
    
    nodes = list(Tree.nodes())
    w = nx.get_node_attributes(Tree,'w')
    x = nx.get_node_attributes(Tree,'x')
    e = nx.get_node_attributes(Tree,'e')
    num_e = {n:len(e[n]) for n in nodes}
    corner = nx.get_node_attributes(Tree,'corner')
    colors = ["#d94386",
              "#7edf47",
              "#7045d1",
              "#d9d53d",
              "#c84aca",
              "#63dd8f",
              "#542d84",
              "#bfdf75",
              "#6579cd",
              "#5ba33f",
              "#c685ce",
              "#a3993d",
              "#35274b",
              "#d58b30",
              "#75b2d5",
              "#dd4931",
              "#64dbc8",
              "#903c23",
              "#bcd3c9",
              "#522728",
              "#c4d197",
              "#883a63",
              "#466728",
              "#cf6368",
              "#5a9578",
              "#d79c72",
              "#2e402e",
              "#ca9fb4",
              "#7e6642",
              "#566a83"]
    # colors = ['#e6194b',
    #           '#3cb44b',
    #           '#ffe119',
    #           '#0082c8',
    #           '#f58231',
    #           '#911eb4',
    #           '#46f0f0',
    #           '#f032e6',
    #           '#d2f53c',
    #           '#fabebe',
    #           '#008080',
    #           '#e6beff',
    #           '#aa6e28',
    #           '#fffac8',
    #           '#800000',
    #           '#aaffc3',
    #           '#808000',
    #           '#ffd8b1',
    #           '#000080',
    #           '#808080',
    #           '#ffffff',
    #           '#000000']
    nb_col=0
    assoc_colors={-1:'#000000'}
    for n in Tree.node[0]["e"]:
        assoc_colors[n]=colors[nb_col]
        nb_col+=1
	
    nN = len(nodes)
    index = {nodes[i]:i for i in range(nN)}
    # ind_r=index[str(set(G.nodes()))] # root's index
	
    G_0=gt.Graph(directed=False)
    v_gpe_id = G_0.new_vertex_property("string") #node ID
    v_size=G_0.new_vertex_property("int")
    v_threshold=G_0.new_vertex_property("int")
    v_pos=G_0.new_vertex_property("int")
    v_frac=G_0.new_vertex_property("vector<double>")
    v_col=G_0.new_vertex_property("vector<string>")
    e_checkp=G_0.new_edge_property("vector<double>")
    v_gpe_list=[]

    dd=3.5 #size of the curved angle
    de=10. #size of the depth increment
    
    lLeaf = []
    for n in nodes:
            v = G_0.add_vertex()
            if (num_e[n] == 1)and(n != -1):
                    lLeaf.append(v)
                    v_gpe_id[v] = e[n][0]
            else:
                    v_gpe_id[v] = ""
            v_gpe_list.append(v)
            v_threshold[v]=w[n]
            v_pos[v]=x[n]*de
            if n == -1:
                v_size[v]=5
            else:
                v_size[v]=20
            v_frac[v]=[1./num_e[n] for k in range(num_e[n])]
            v_col[v]=[assoc_colors[n_e] for n_e in e[n]]
	
    for n,p in Tree.edges():
            if num_e[n] < num_e[p]:
                    n,p = p,n
            i,j = index[n],index[p]
            e = G_0.add_edge(v_gpe_list[i],v_gpe_list[j])

            thA, posA=float(w[n]),float(x[n])*de
            thB, posB=float(w[p]),float(x[p])*de
            if posB-posA !=0:
                    rAB=np.sqrt(((posB-posA)**2)+((thB-thA)**2))
                    alpha=np.arctan(abs(thB-thA)/abs(posB-posA))
                    rc=np.cos(alpha)*(abs(posB-posA))
                    yc=np.sqrt(((abs(posB-posA))**2)-rc**2)

                    s1 = abs(posB-posA)/(abs(posB-posA)-dd)
                    rc1=rc/s1
                    yc1=yc/s1
                    s2 = abs(thB-thA)/(abs(thB-thA)-dd)
                    rc2=rAB - (rAB-rc)/s2
                    yc2=yc/s2
                    e_checkp[e]=[0,0,0,0,rc1/rAB,yc1,rc1/rAB,yc1,rc/rAB,yc,rc/rAB,yc,rc2/rAB,yc2,rc/rAB,yc,1,0,1,0]
    G_0.vertex_properties["ID"] = v_gpe_id

    pos_gpe = arf_layout(G_0)
#    v_gpe_x = G_0.new_vertex_property("string") # node x coordinates
#    v_gpe_y = G_0.new_vertex_property("string") # node x coordinates

    # SET DEPTH & THRESHOLD
    tha=np.zeros((2,len(nodes)))
    tha[0]=list(v_pos)
    tha[1]=list(v_threshold)
    pos_gpe.set_2d_array(tha,[0,1])
    
#    for vk in range(len(v_gpe_list)):
#            val='%d ' % pos_gpe.get_2d_array([0])[0][vk]
#            val+=str(v_gpe_id[vk])
#            v_gpe_x[v_gpe_list[vk]] = val

    m = 10.
    s = 10.
    dx = depth*de+2.*s+m
    ymax = int(max(thW))+10
    dy = ymax+2.*s+2.*m
    ratio = dx/dy
    H = 10.
    L = H*ratio
    fig=plt.figure(figsize=(L,H))
    ax=fig.add_axes([m/dx,m/dy,(dx-m)/dx,(dy-2.*m)/dy])

    graph_draw(G_0, pos_gpe, output_size=(1000,1000),
                    nodesfirst=True,
                    edge_marker_size=10,
                    edge_pen_width=1,
                    edge_control_points=e_checkp,
                    vertex_size=v_size,
                    vertex_shape='pie',
                    vertex_pie_fractions=v_frac,
                    vertex_pie_colors=v_col,
                    #vertex_fill_color=colors[gp],
                    #vertex_text=v_gpe_id,
                    #bg_color=[1., 1., 1., 1.],
                    #output=d+"/tree_"+str(gp)+"_"+d+".png"
                    mplfig=ax)

    for v in lLeaf:
            ax.text(v_pos[v],ymax+10,v_gpe_id[v],horizontalalignment='center',verticalalignment='center')
    if n_ta >= 0:
        ax.hlines(th_ta,-s,depth*de+s,colors='r',linestyles='dashed')

    ax.set_xlim(-s,depth*de+s)
    ax.set_ylim(ymax+s,-s)
    ax.set_xticklabels([])
    ax.set_yticks(list(range(0,ymax+10,10)))
    ax.get_xaxis().set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    #----ticks
    ax.yaxis.set_ticks_position('left')
    plt.savefig(name)
    plt.close(fig)
#------------------------------------------
#------------------------------------------
#==========================================
#==========================================
#------------------------------------------
depth = 0
"""