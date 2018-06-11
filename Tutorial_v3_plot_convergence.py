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
    "l_ptau":{"TL":0,"k":1,"s":0,"w":0,"n":0,"tau":1,"dtau":0,"a":0,"alpha":0,"dalpha":0}
}

def convplot(ax,f,n,data_n):
    if Const[f][n]:
        ax.plot([.5],[.5],"bx",ms=50,mew=3)
    else:
        ax.plot(data_n,'r-',lw=3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim((0,0.99))
    if n == "tau":
        ax.set_ylabel(r"$p(\tau)$")
    elif n == "dtau":
        ax.set_ylabel(r"$p(\Delta\tau)$")
    elif n == "alpha":
        ax.set_ylabel(r"$p(\alpha)$")
    elif n == "dalpha":
        ax.set_ylabel(r"$p(\Delta\alpha)$")
    else:
        ax.set_ylabel("$p("+n+")$")
    ax.yaxis.set_label_coords(.15,.87)

def compare(list_methods,filename):
    entree = open("analysis_init.dat")
    data_init = load(entree)
    entree.close()

    #plot measures
    a = 2.
    m = 0.5
    nX = 10
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
        entree = open("conv_"+f+".dat")
        data = load(entree)
        entree.close()
        #--activity timeline
        ax = fig.add_axes([x,y,w,h])
        if Const[f]["TL"]:
            ax.plot([.5],[.5],"bx",ms=50,mew=3)
        else:
            ax.plot(data["data_TL"],'r-',lw=2)
        ax.set_xticks([])
#        ax.set_yticks([])
        ax.set_ylim((0.01,1))
        if i == nY-1:
            ax.set_xlabel("$t$")
        ax.set_ylabel("$E_t$")
        ax.yaxis.set_label_coords(.15,.9)
        #--degrees
        x += w
        ax = fig.add_axes([x,y,w,h])
        convplot(ax,f,"k",data["data_k"])
        if i == nY-1:
            ax.set_xlabel("$k$")
        #--strength
        x += w
        ax = fig.add_axes([x,y,w,h])
        ax.set_ylim((0,1))
        convplot(ax,f,"s",data["data_s"])
        if i == nY-1:
            ax.set_xlabel("$s$")
        #--weights
        x += w
        ax = fig.add_axes([x,y,w,h])
        convplot(ax,f,"w",data["data_w"])
        if i == nY-1:
            ax.set_xlabel("$w$")
        #--number of contacts
        x += w
        ax = fig.add_axes([x,y,w,h])
        convplot(ax,f,"n",data["data_n"])
        if i == nY-1:
            ax.set_xlabel("$n$")
        #--contact durations
        x += w
        ax = fig.add_axes([x,y,w,h])
        convplot(ax,f,"tau",data["data_tau"])
        if i == nY-1:
            ax.set_xlabel(r"$\tau$")
        #--intercontact durations
        x += w
        ax = fig.add_axes([x,y,w,h])
        convplot(ax,f,"dtau",data["data_dtau"])
        if i == nY-1:
            ax.set_xlabel(r"$\Delta\tau$")
        #--activity
        x += w
        ax = fig.add_axes([x,y,w,h])
        convplot(ax,f,"a",data["data_a"])
        if i == nY-1:
            ax.set_xlabel("$a$")
        #--activity durations
        x += w
        ax = fig.add_axes([x,y,w,h])
        convplot(ax,f,"alpha",data["data_alpha"])
        if i == nY-1:
            ax.set_xlabel(r"$\alpha$")
        #--inactivity durations
        x += w
        ax = fig.add_axes([x,y,w,h])
        convplot(ax,f,"dalpha",data["data_dalpha"])
        if i == nY-1:
            ax.set_xlabel(r"$\Delta\alpha$")

        ax.text(xt,yt,"P__"+f,rotation=-90,transform=ax.transAxes,horizontalalignment='center',verticalalignment='center'
)
        #changement de ligne
        y -= h
        x = xo

    #save
    plt.savefig(filename)

list1 = ["pTheta_n","pTheta_Gstat","pTheta_kstat","pTheta","t_delta"]
list2 = ["pitau_pidtau_t1","pitau_pidtau","pitau_t1_tw","pitau","delta_Gstat"]

compare(list1,"conv_link_v3.png")
compare(list2,"conv_time_v3.png")
