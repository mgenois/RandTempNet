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
from sys import argv
from glob import glob
from copy import deepcopy
from math import ceil
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path
import matplotlib.lines as lines
#------------------------------------------
#Colours
day_col = 'r'
c_col = "#ff8080"
l_col = "#6666ff"
g_col = "#33cc33"
fo_col = "#ccccff"
fi_col = "#adebad"
#------------------------------------------
#==========================================
#==========================================
#------------------------------------------
#Computing link flows between separate tij files
#------------------------------------------
#Extraction of the lists of graphs, edge lists and node lists
#-path: path to the folder containing the tij files (with final "/")
def extract_lists(path):
    lFiles = glob(path+"*")
    lFiles.sort()
    lGraph = []
    lEdges = []
    lNodes = []
    for f in lFiles:
        data = np.loadtxt(f,dtype='int')
        G_d = nx.Graph()
        for c in data:
            n,p = c[1],c[2]
            if G_d.has_edge(n,p):
                G_d[n][p]['w'] += 1.
            else:
                G_d.add_edge(n,p,w=1.)
        G = set([(min(n,p),max(n,p)) for n,p in G_d.edges()])
        lGraph.append(G_d)
        lEdges.append(G)
        lNodes.append(G_d.nodes())
    return lGraph,lEdges,lNodes
#------------------------------------------
#Analysis of the flows of links
#-lEdges: list of the list of edges per day
#-lNodes: list of the list of nodes per day
def analysis_flows(lEdges,lNodes):
    nD = len(lEdges)
    #------------------------------------------
    #out
    Flow_out = [] #links connected to exiting nodes
    Cons = [] #links transmitted
    Lost = [] #links lost
    for d in range(nD-1):
        #extraction of the common links
        E = lEdges[d]&lEdges[d+1]
        Cons.append(E)
        lRest = lEdges[d] - E
        #sorting between links lost and links connected to lost nodes
        E = []
        for n,p in lRest:
            if (n in lNodes[d+1])and(p in lNodes[d+1]):
                E.append((n,p))
        E = set(E)
        Lost.append(E)
        Flow_out.append(lRest - E)
    Flow_out.append(set([]))
    Cons.append(set([]))
    Lost.append(set([]))
    #------------------------------------------
    #in
    Flow_in = [set([])] #links connected to entering nodes
    Gain = [set([])] #links gained
    for d in range(1,nD):
        #extraction of the common links
        E = lEdges[d]&lEdges[d-1]
        lRest = lEdges[d] - E
        #sorting between links gained and links connected to entering nodes
        E = []
        for n,p in lRest:
            if (n in lNodes[d-1])and(p in lNodes[d-1]):
                E.append((n,p))
        E = set(E)
        Gain.append(E)
        Flow_in.append(lRest - E)
    #------------------------------------------
    return (Flow_out,Flow_in,Cons,Lost,Gain)
#------------------------------------------
#------------------------------------------
#Print of the flows calculated by analysis_flows
def count_flows(lEdges,Flow_out,Flow_in,Cons,Lost,Gain,check=False):
    nD = len(Cons)
    #------------------------------------------
    #numbers
    nE = [len(e) for e in lEdges]
    nFlow_out = [len(e) for e in Flow_out]
    nFlow_in = [len(e) for e in Flow_in]
    nCons = [len(e) for e in Cons]
    nLost = [len(e) for e in Lost]
    nGain = [len(e) for e in Gain]
    #check
    if check:
        for d in range(nD):
            #first day
            if d == 0:
                if nE[d] != nFlow_out[d] + nLost[d] + nCons[d]:
                    print "Error count out (Day %d): %d != %d + %d + %d" % (d,nE[d],nFlow_out[d],nLost[d],nCons[d])
                else:
                    print "Count out (Day %d): OK" % d
            #last day
            elif d == nD-1:
                if nE[d] != nFlow_in[d] + nGain[d] + nCons[d-1]:
                    print "Error count in (Day %d): %d != %d + %d + %d" % (d,nE[d],nFlow_in[d],nGain[d],nCons[d-1])
                else:
                    print "Count in (Day %d): OK" % d
            #other days
            else:
                if nE[d] != nFlow_out[d] + nLost[d] + nCons[d]:
                    print "Error count out (Day %d): %d != %d + %d + %d" % (d,nE[d],nFlow_out[d],nLost[d],nCons[d])
                else:
                    print "Count out (Day %d): OK" % d
                if nE[d] != nFlow_in[d] + nGain[d] + nCons[d-1]:
                    print "Error count in (Day %d): %d != %d + %d + %d" % (d,nE[d],nFlow_in[d],nGain[d],nCons[d-1])
                else:
                    print "Count in (Day %d): OK" % d
    return (nE,nFlow_out,nFlow_in,nCons,nLost,nGain)
#------------------------------------------
#------------------------------------------
#Plot of the flows calculated by analysis_flows
def plot_flows(nE,nFlow_out,nFlow_in,nCons,nLost,nGain):
    #------------------------------------------
    #figsize measures
    a = 4. #height of the main day
    b = 5. #inter-day spacing
    m = 0.5 #margin
    l = 0.5 #width of one day
    r = 0.5 #inner radius of the arrows
    #------------------------------------------
    nD = len(nE)
    Ha = 2.*m + a + 2.*r
    Lo = 2.*m + (nD-1)*b + nD*l
    mx = m/Lo
    my = m/Ha
    w = (Lo - 2.*m)/Lo
    h = (Ha - 2.*m)/Ha

    lx = l/a
    bx = b/a + lx
    dr = r/a
    norm = float(max(nE))
    
    #plot
    fig = plt.figure(figsize=(Lo,Ha))
    axe = fig.add_axes([mx,my,w,h])
    plt.axis('equal')
    for d in range(nD):
        x = d*bx
        ze = nE[d]/norm
        #--flow_out
        z = nFlow_out[d]/norm
        p = patches.Wedge(
            (x+lx,-dr), #center
            dr+z, #radius
            0, #theta1
            90, #theta2
            width = z,
            facecolor=fo_col,
            edgecolor="none"
        )
        axe.add_patch(p)
        l = lines.Line2D([x+lx,x+lx+z/3.,x+lx],
                         [0,z/2.,z],
                         lw=2.,
                         color='w'
        )
        axe.add_line(l)
        #--flow_in
        z = nFlow_in[d]/norm
        p = patches.Wedge(
            (x,-dr), #center
            dr+z, #radius
            90, #theta1
            180, #theta2
            width = z,
            facecolor=fi_col,
            edgecolor="none"
        )
        axe.add_patch(p)
        l = lines.Line2D([x-dr,x-dr-z/2.,x-dr-z],
                         [-dr,-dr+z/3.,-dr],
                         lw=2.,
                         color='w'
        )
        axe.add_line(l)
        #--lost links
        z = nLost[d]/norm
        p = patches.Wedge(
            (x+lx,ze+dr), #center
            dr+z, #radius
            -90, #theta1
            0, #theta2
            width = z,
            facecolor=l_col,
            edgecolor="none"
        )
        axe.add_patch(p)
        l = lines.Line2D([x+lx,x+lx+z/3.,x+lx],
                         [ze,ze-z/2.,ze-z],
                         lw=2.,
                         color='w'
        )
        axe.add_line(l)
        #--gained links
        z = nGain[d]/norm
        p = patches.Wedge(
            (x,ze+dr), #center
            dr+z, #radius
            180, #theta1
            -90, #theta2
            width = z,
            facecolor=g_col,
            edgecolor="none"
        )
        axe.add_patch(p)
        l = lines.Line2D([x-dr,x-dr-z/2.,x-dr-z],
                         [ze+dr,ze+dr-z/3.,ze+dr],
                         lw=2.,
                         color='w'
        )
        axe.add_line(l)
        #--conserved links
        if d < nD-1:
            zc = nCons[d]/norm
            zo = nFlow_out[d]/norm
            zi = nFlow_in[d+1]/norm
            Path = path.Path
            path_data = [
                (Path.MOVETO,(x+lx,zo)),
                (Path.LINETO,(x+lx,zo+zc)),
                (Path.CURVE4,(x+bx/2.,zo+zc)),
                (Path.CURVE4,(x+bx/2.,zi+zc)),
                (Path.CURVE4,(x+bx,zi+zc)),
                (Path.LINETO,(x+bx,zi)),
                (Path.CURVE4,(x+bx/2.,zi)),
                (Path.CURVE4,(x+bx/2.,zo)),
                (Path.CURVE4,(x+lx,zo)),
                (Path.CLOSEPOLY,(x+lx,zo))
            ]
            codes, verts = zip(*path_data)
            q = path.Path(verts,codes)
            p = patches.PathPatch(
                q,
                facecolor=c_col,
                edgecolor="none",
            )
            axe.add_patch(p)
        #--day
        p = patches.Rectangle(
            (x,0),
            lx,
            ze,
            facecolor=day_col,
            edgecolor="none"
        )
        axe.add_patch(p)
        axe.text(x+lx/2.,-dr/2.,d+1,fontsize=20,horizontalalignment='center',verticalalignment='center')
    axe.set_xlim((0,x+lx))
    axe.set_ylim((-dr,1+dr))
    axe.axis("off")
    plt.savefig("flows.pdf")
    plt.close(fig)
#------------------------------------------
#------------------------------------------
#Analysis of the weight evolution
#-lGraph: list of the graphs
#-E     : list of links
#-d1    : index of the first day
#-d2    : index of the second day
def analysis_weight(lGraph,E,d1,d2):
    G1 = lGraph[d1]
    G2 = lGraph[d2]
    
    evol = [G2[n][p]['w']/G1[n][p]['w'] for n,p in E]

    n = len(evol)
    n_plus = float(len([e for e in evol if e >= 1]))
    n_moins = float(n - n_plus)
    n_plus /= n
    n_moins /= n

    return evol,n_plus,n_moins
#------------------------------------------
#------------------------------------------
#Plot of the weight evolution
#-lGraph: list of the graphs
#-lEdges: list of the list of edges
#-lC: list of the conserved links per day
def plot_evol(lGraph,lEdges,lC):
    List_Col = ['#006600', '#00ff99', '#cc6633', '#9900cc', '#339999', '#ffcc00', '#330066', '#003366', '#666666', '#99ccff', '#cccc00', '#cccccc', '#ff6666', '#cc99cc', '#993300', '#666600', '#339966', '#3399ff', '#009999', '#ff6699', '#000099', '#cc99ff', '#0066cc', '#33cc33', '#ff6600', '#990099', '#ff9900', '#336666', '#ffff00', '#999900', '#66ff00', '#cc3300', '#3300ff', '#996600', '#663399', '#cc3366', '#000000', '#990066', '#ffcc99', '#99cc00', '#ccff00', '#ff0000', '#996666']
    db = 1.

    nD = len(lEdges)
    mx = 0.15/(nD-1)
    w = 0.82/(nD-1)
    lEvol = []
    fig1 = plt.figure(1,figsize=(5,5))
    ax = fig1.add_axes([0.15,0.15,0.82,0.82])
    fig2 = plt.figure(2,figsize=(5*(nD-1),5))
    for d in range(nD-1):
        data,npl,nmo = analysis_weight(lGraph,lC[d],d,d+1)
        min_val = np.log2(min(data))
        max_val = np.log2(max(data))
        x = 2**np.arange(min_val-db,max_val+db,db)
        hist = np.histogram(data,bins = x,density = True)
        #------------------------------------------
        ax_loc = fig2.add_axes([d/(nD-1.)+mx,0.15,w,0.79])
        ax_loc.loglog(x[1:],hist[0],'o',mec=List_Col[d],mfc="none",label=str(d+1)+r"$\rightarrow$"+str(d+2))
        ax_loc.set_xlabel("$r = w_{i+1}/w_i$",fontsize=15)
        ax_loc.set_ylabel("$p(r)$",fontsize=15)
        ax_loc.set_xlim([0.01,100])
        ax_loc.set_ylim([0.0001,10])
        ax_loc.text(0.1,0.001,"%.3f" % nmo,fontsize=15)
        ax_loc.text(10,1,"%.3f" % npl,fontsize=15)
        y1,y2 = ax_loc.get_ylim()
        ax_loc.vlines([1],y1,y2,linestyles='dotted')
        ax_loc.set_title(str(d+1)+r"$\rightarrow$"+str(d+2),fontsize=20)
        #------------------------------------------
        ax.loglog(x[1:],hist[0],'o',mec=List_Col[d],mfc="none",label=str(d+1)+r"$\rightarrow$"+str(d+2))
        lEvol += data
    plt.savefig("evol_d.pdf")
    plt.close(fig2)

    min_val = np.log2(min(lEvol))
    max_val = np.log2(max(lEvol))
    x = 2**np.arange(min_val-db,max_val+db,db)
    hist = np.histogram(lEvol,bins = x,density = True)
    ax.loglog(x[1:],hist[0],'+',mec="r",mfc="none",mew=2,ms=10,label="Global")
    y1,y2 = ax.get_ylim()
    ax.vlines([1],y1,y2,linestyles='dotted')

    np.savetxt("evol.dat",lEvol)

    n = len(lEvol)
    npl = float(len([e for e in lEvol if e >= 1]))
    nmo = float(n - npl)
    npl /= n
    nmo /= n
    ax.text(0.1,0.001,"%.3f" % nmo,fontsize=15,horizontalalignment="center")
    ax.text(10,0.001,"%.3f" % npl,fontsize=15,horizontalalignment="center")

    ax.legend(loc=0,ncol=2,numpoints=1,fontsize=9)
    ax.set_xlabel("$r = w_{i+1}/w_i$",fontsize=15)
    ax.set_ylabel("$p(r)$",fontsize=15)
    plt.savefig("evol.pdf")
    plt.close(fig1)
#-----------------------------------------------------------
#-----------------------------------------------------------
#Print of the numbers calculated by count_flows
def print_flows(nEdges,nFlow_out,nFlow_in,nCons,nLost,nGain):
    nD = len(nEdges)
    sortie = open("flows.dat",'w')
    #-----------------------------------------------------------
    sortie.write("Day")
    for d in range(nD):
        sortie.write("\t"+str(d))
    sortie.write("\n")
    lLbl = ["Edges","Flow out","Flow in","Conserved","Lost","Gained"]
    for i,liste in enumerate([nEdges,nFlow_out,nFlow_in,nCons,nLost,nGain]):
        sortie.write(lLbl[i])
        for d in range(nD):
            sortie.write("\t"+str(liste[d]))
        sortie.write("\n")
    sortie.close()
#-----------------------------------------------------------
#-----------------------------------------------------------
#Calculates the fraction of links per category for each bin of weight
def flow_probability(lGraph,lFlow_out,lFlow_in,lCons,lLost,lGain):
    nD = len(lGraph)
    #-----------------------------------------------------------
    lN,lCFo,lCFi,lCC1,lCC2,lCL,lCG = [],[],[],[],[],[],[]
    lX = []
    for d in range(nD):
        #extraction of the edges weights
        G = lGraph[d]
        dW = nx.get_edge_attributes(G,'w')
        lW = dW.values()
        lE = G.edges()
        #binning of the links
        min_val = np.log2(min(lW))
        max_val = np.log2(max(lW))
        x = 2**np.arange(min_val,max_val+1)
        lX.append(x)
        nX = len(x)
        norm = np.zeros(nX)
        catFo = np.zeros(nX)
        catFi = np.zeros(nX)
        catC1 = np.zeros(nX)
        catC2 = np.zeros(nX)
        catL = np.zeros(nX)
        catG = np.zeros(nX)
        for n,p in lE:
            i = int(ceil(np.log2(G[n][p]['w'])))
            norm[i] += 1.
        for n,p in lFlow_out[d]:
            i = int(ceil(np.log2(G[n][p]['w'])))
            catFo[i] += 1.
        for n,p in lFlow_in[d]:
            i = int(ceil(np.log2(G[n][p]['w'])))
            catFi[i] += 1.
        for n,p in lCons[d]:
            i = int(ceil(np.log2(G[n][p]['w'])))
            catC1[i] += 1.
        #weights of the conserved links on the following day
        if d > 0:
            for n,p in lCons[d-1]:
                i = int(ceil(np.log2(G[n][p]['w'])))
                catC2[i] += 1.
        for n,p in lLost[d]:
            i = int(ceil(np.log2(G[n][p]['w'])))
            catL[i] += 1.
        for n,p in lGain[d]:
            i = int(ceil(np.log2(G[n][p]['w'])))
            catG[i] += 1.
        lN.append(norm)
        lCFo.append(catFo)
        lCFi.append(catFi)
        lCC1.append(catC1)
        lCC2.append(catC2)
        lCL.append(catL)
        lCG.append(catG)
    #plot
    mx = 0.15/(nD-1)
    w = 0.82/(nD-1)
    fsize = 20
    fig = plt.figure(1,figsize=(5*(nD-1),5))
    for d in range(nD-1):
        ax = fig.add_axes([d/(nD-1.)+mx,0.15,w,0.79])
#        b = np.zeros(nX)
#        ax.bar(lX[d],lCFo[d]/lN[d],width=lX[d],color=fo_col,edgecolor='none')
#        b += lCFo[d]/lN[d]
#        ax.bar(lX[d],lCC1[d]/lN[d],bottom=b,width=lX[d],color=c_col,edgecolor='none')
#        b += lCC1[d]/lN[d]
#        ax.bar(lX[d],lCL[d]/lN[d],bottom=b,width=lX[d],color=l_col,edgecolor='none')
        ax.stackplot(lX[d],lCFo[d]/lN[d],lCC1[d]/lN[d],lCL[d]/lN[d],colors=[fo_col,c_col,l_col])
        ax.set_xscale('log')
        ax.set_xlim((lX[d][0],lX[d][-1]))
        ax.set_xlabel("$w$",fontsize=fsize)
        ax.set_ylabel("$f$",fontsize=fsize)
        ax.set_title(str(d+1)+r"$\rightarrow$"+str(d+2),fontsize=20)
    plt.savefig("prob_ex.pdf")
    plt.close(fig)
    fig = plt.figure(2,figsize=(5*(nD-1),5))
    for d in range(1,nD):
        ax = fig.add_axes([(d-1)/(nD-1.)+mx,0.15,w,0.79])
        ax.stackplot(lX[d],lCFi[d]/lN[d],lCC2[d]/lN[d],lCG[d]/lN[d],colors=[fi_col,c_col,g_col])
        ax.set_xscale('log')
        ax.set_xlim((lX[d][0],lX[d][-1]))
        ax.set_xlabel("$w$",fontsize=fsize)
        ax.set_ylabel("$f$",fontsize=fsize)
        ax.set_title(str(d)+r"$\rightarrow$"+str(d+1),fontsize=20)
    plt.savefig("prob_in.pdf")
    plt.close(fig)
#-----------------------------------------------------------
#-----------------------------------------------------------
#Function to do everything at once
def compute_all(path):
    lGraph,lEdges,lNodes = extract_lists(path)
    lFo,lFi,lC,lL,lG = analysis_flows(lEdges,lNodes)
    nE,nFo,nFi,nC,nL,nG = count_flows(lEdges,lFo,lFi,lC,lL,lG)
    print_flows(nE,nFo,nFi,nC,nL,nG)
    plot_flows(nE,nFo,nFi,nC,nL,nG)
    plot_evol(lGraph,lEdges,lC)
    flow_probability(lGraph,lFo,lFi,lC,lL,lG)
#------------------------------------------
#==========================================
#==========================================
#------------------------------------------
