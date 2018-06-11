import itertools as it
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from pickle import load

Const = {
    "n_pTheta":{"TL":1,"k":1,"s":0,"w":1,"n":1,"tau":1,"dtau":1,"a":1,"alpha":0,"dalpha":0},
    "L_pTheta":{"TL":1,"k":1,"s":0,"w":1,"n":1,"tau":1,"dtau":1,"a":0,"alpha":0,"dalpha":0},
    "pTheta_k":{"TL":1,"k":1,"s":0,"w":1,"n":1,"tau":1,"dtau":1,"a":0,"alpha":0,"dalpha":0},
    "pTheta":{"TL":1,"k":0,"s":0,"w":1,"n":1,"tau":1,"dtau":1,"a":0,"alpha":0,"dalpha":0},
    "pttau":{"TL":1,"k":0,"s":0,"w":0,"n":0,"tau":1,"dtau":0,"a":0,"alpha":0,"dalpha":0},
    "pitau_pidtau_t1":{"TL":0,"k":1,"s":1,"w":1,"n":1,"tau":1,"dtau":1,"a":1,"alpha":0,"dalpha":0},
    "pitau_pidtau":{"TL":0,"k":1,"s":1,"w":1,"n":1,"tau":1,"dtau":1,"a":1,"alpha":0,"dalpha":0},
    "pitau_t1_tw":{"TL":0,"k":1,"s":1,"w":1,"n":1,"tau":1,"dtau":0,"a":1,"alpha":0,"dalpha":0},
    "pitau":{"TL":0,"k":1,"s":1,"w":1,"n":1,"tau":1,"dtau":0,"a":1,"alpha":0,"dalpha":0},
    "L_ptau":{"TL":0,"k":1,"s":0,"w":0,"n":0,"tau":1,"dtau":0,"a":0,"alpha":0,"dalpha":0}
}

def condplot(ax,n,f,data_init_n,data_n):
    if n == "k":
        max_k = max(data_init_n)+5
        bins = range(0,max_k,5)
    elif (n == "a")or(n == "s"):
        max_n = max(data_init_n)+0.25
        bins = 2.**(np.arange(0,np.log2(max_n),0.25))
    else:
        max_n = max(data_init_n)+0.5
        bins = 2.**(np.arange(0,np.log2(max_n),0.5))
    dist = np.histogram(data_init_n,bins=bins,density=True)
    ax.plot(dist[1][:-1],dist[0],'o',mec='k',mew=1.5,mfc='None')
    
    bins = data_n["bins"]
    if Const[f][n]:
        ax.plot(bins[:-1],data_n["med"],'^',mec='b',mew=1.5,mfc='None')
    else:
        ax.fill_between(bins[:-1],data_n["inf"],data_n["sup"],color='r',lw=0,alpha=0.3)
        ax.plot(bins[:-1],data_n["med"],'o',mec='r',mew=1.5,mfc='None')

def compare(list_methods,filename):
    entree = open("analysis_init.dat")
    data_init = load(entree)
    entree.close()

    #plot measures
    a = 2.
    m = 0.5
    nX = 11
    nY = len(list_methods)
    LX = nX*a+2*m
    LY = nY*a+2*m
    fig = plt.figure(figsize=(LX,LY))
    r = LX/LY
    xo = m/LX
    xf = xo
    yo = r*xo
    yf = yo
    w = a/LX
    h = r*w
    x = xo
    y = 1. - yf - h
    xt = 1.06
    yt = 0.5
    for i,f in enumerate(list_methods):
        entree = open("med_"+f+".dat")
        data = load(entree)
        entree.close()
        #--activity timeline
        w_tl = 2*w
        ax = fig.add_axes([x,y,w_tl,h])
        ax.plot(data_init["data_TL"],'k-')
        xmax = len(data_init["data_TL"])
        if Const[f]["TL"]:
            ax.plot(data["data_TL"]["med"],'b-')
        else:
            ax.plot(data["data_TL"]["med"],'r-',lw=2)
            ax.fill_between(range(xmax),data["data_TL"]["inf"],data["data_TL"]["sup"],color='r',alpha=0.3)
        ax.set_xlim((0,xmax))
        ax.set_xticks([])
        ax.set_yticks([])
        if i == nY-1:
            ax.set_xlabel("$t$")
        ax.set_ylabel("$E_t$",labelpad=-20)
        #--degrees
        x += w_tl
        ax = fig.add_axes([x,y,w,h])
        condplot(ax,"k",f,data_init["data_k"],data["data_k"])
        ax.set_xticks([])
        ax.set_yticks([])
        if i == nY-1:
            ax.set_xlabel("$k$")
        ax.set_ylabel("$p(k)$",labelpad=-20)
        #--strength
        x += w
        ax = fig.add_axes([x,y,w,h])
        condplot(ax,"s",f,data_init["data_s"],data["data_s"])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim((500,100000))
        if i == nY-1:
            ax.set_xlabel("$s$",labelpad=0)
        ax.set_ylabel("$p(s)$",labelpad=-23)
        #--weights
        x += w
        ax = fig.add_axes([x,y,w,h])
        condplot(ax,"w",f,data_init["data_w"],data["data_w"])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim((5,20000))
        if i == nY-1:
            ax.set_xlabel("$w$",labelpad=0)
        ax.set_ylabel("$p(w)$",labelpad=-23)
        #--number of contacts
        x += w
        ax = fig.add_axes([x,y,w,h])
        condplot(ax,"n",f,data_init["data_n"],data["data_n"])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim((0.5,400))
        ax.set_ylim((0.000001,2))
        if i == nY-1:
            ax.set_xlabel("$n$",labelpad=0)
        ax.set_ylabel("$p(n)$",labelpad=-23)
        #--contact durations
        x += w
        ax = fig.add_axes([x,y,w,h])
        condplot(ax,"tau",f,data_init["data_tau"],data["data_tau"])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim((10,2000))
        if i == nY-1:
            ax.set_xlabel(r"$\tau$",labelpad=0)
        ax.set_ylabel(r"$p(\tau)$",labelpad=-23)
        #--intercontact durations
        x += w
        ax = fig.add_axes([x,y,w,h])
        condplot(ax,"dtau",f,data_init["data_dtau"],data["data_dtau"])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim((5,200000))
        ax.set_ylim((0.00000001,0.1))
        if i == nY-1:
            ax.set_xlabel(r"$\Delta\tau$",labelpad=0)
        ax.set_ylabel(r"$p(\Delta\tau)$",labelpad=-23)
        #--activity
        x += w
        ax = fig.add_axes([x,y,w,h])
        condplot(ax,"a",f,data_init["data_a"],data["data_a"])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim((50,2000))
        ax.set_ylim((0.00005,0.01))
        if i == nY-1:
            ax.set_xlabel("$a$",labelpad=0)
        ax.set_ylabel("$p(a)$",labelpad=-23)
        #--activity durations
        x += w
        ax = fig.add_axes([x,y,w,h])
        condplot(ax,"alpha",f,data_init["data_alpha"],data["data_alpha"])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim((0.5,300))
        ax.set_ylim((0.00000001,10))
        if i == nY-1:
            ax.set_xlabel(r"$\alpha$",labelpad=0)
        ax.set_ylabel(r"$p(\alpha)$",labelpad=-23)
        #--inactivity durations
        x += w
        ax = fig.add_axes([x,y,w,h])
        condplot(ax,"dalpha",f,data_init["data_dalpha"],data["data_dalpha"])
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim((10,100000))
        if i == nY-1:
            ax.set_xlabel(r"$\Delta\alpha$",labelpad=0)
        ax.set_ylabel(r"$p(\Delta\alpha)$",labelpad=-23)

        ax.text(xt,yt,"P__"+f,rotation=-90,transform=ax.transAxes,horizontalalignment='center',verticalalignment='center'
)
        #changement de ligne
        y -= h
        x = xo

    #save
    plt.savefig(filename)

list1 = ["n_pTheta","L_pTheta","pTheta_k","pTheta","pttau"]
list2 = ["pitau_pidtau_t1","pitau_pidtau","pitau_t1_tw","pitau","L_ptau"]

compare(list1,"analyse_link_v3.png")
compare(list2,"analyse_time_v3.png")
