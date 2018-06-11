#computes the convergence of the median measures when adding randomisations

from glob import glob
from pickle import load,dump
from scipy.stats import entropy
from numpy.linalg import norm
import numpy as np

def JSD(P, Q):
    _P = P / norm(P, ord=1)
    _Q = Q / norm(Q, ord=1)
    _M = 0.5 * (_P + _Q)
    return 0.5 * (entropy(_P, _M) + entropy(_Q, _M))

def distrib_init(data_in,d):
    data_n = []
    max_n = max(data_in)+d
    bins_n = 2.**(np.arange(0,np.log2(max_n),d))
    dist = np.histogram(data_in,bins=bins_n,density=True)
    for n in dist[0]:
        data_n.append([n])
    return data_n,max_n,bins_n

def distrib_fill(data_in,data_n,max_n,bins_n,d):
    m = max(data_in)+d
    if m > max_n:
        #extension de la distribution
        max_n = m
        l1 = len(bins_n)-1
        bins_n = 2.**(np.arange(0,np.log2(max_n),d))
        l2 = len(bins_n)-1
        fill = [0 for n in data_n[0]]
        for i in range(l1,l2):
            data_n.append(fill[:])
    dist = np.histogram(data_in,bins=bins_n,density=True)
    for i,n in enumerate(dist[0]):
        data_n[i].append(n)
    return max_n,bins_n

def distrib_analysis(data_n,bins_n,data_init,bins_init):
    med_n = [np.median(n) for n in data_n]
    l1 = len(bins_n)
    l2 = len(bins_init)
    ref = list(data_init)
    if l1 > l2:
        ref += [0 for n in range(l2,l1)]
    elif l1 < l2:
        med_n += [0 for n in range(l1,l2)]
    return JSD(med_n,ref)

def med_std(m,data_init):
    #liste des fichiers
    list_f = glob("Tuto_analysis/"+m+"/*")

    #initialisation des listes
    print "Init..."
    f = list_f.pop()
    entree = open(f)
    data = load(entree)
    #--activity timeline
    data_TL = []
    for n in data["data_TL"]:
        data_TL.append([n])
    nT = len(data_TL)
    bins_TL = range(nT)
    #--activity
    data_a,max_a,bins_a = distrib_init(data["data_a"],0.25)
    #--activity durations
    data_alpha,max_alpha,bins_alpha = distrib_init(data["data_alpha"],0.5)
    #--inactivity durations
    data_dalpha,max_dalpha,bins_dalpha = distrib_init(data["data_dalpha"],0.5)
    #--degrees
    data_k = []
    max_k = max(data["data_k"])+5
    bins_k = range(0,max_k,5)
    dist = np.histogram(data["data_k"],bins=bins_k,density=True)
    for n in dist[0]:
        data_k.append([n])
    #--strengths
    data_s,max_s,bins_s = distrib_init(data["data_s"],0.25)
    #--weights
    data_w,max_w,bins_w = distrib_init(data["data_w"],0.5)
    #--number of contacts
    data_n,max_n,bins_n = distrib_init(data["data_n"],0.5)
    #--contact durations
    data_tau,max_tau,bins_tau = distrib_init(data["data_tau"],0.5)
    #--intercontact durations
    data_dtau,max_dtau,bins_dtau = distrib_init(data["data_dtau"],0.5)

    #convergence initialisation
    conv_TL = []
    conv_a = []
    conv_alpha = []
    conv_dalpha = []
    conv_k = []
    conv_s = []
    conv_w = []
    conv_n = []
    conv_tau = []
    conv_dtau = []
    print "Init OK"

    #remplissage et convergence
    print "Fill..."
    for j,f in enumerate(list_f):
        entree = open(f)
        data = load(entree)
        #--activity timeline
        for i,n in enumerate(data["data_TL"]):
            data_TL[i].append(n)
        #--activity
        max_a,bins_a = distrib_fill(data["data_a"],data_a,max_a,bins_a,0.25)
        #--activity durations
        max_alpha,bins_alpha = distrib_fill(data["data_alpha"],data_alpha,max_alpha,bins_alpha,0.5)
        #--inactivity durations
        max_dalpha,bins_dalpha = distrib_fill(data["data_dalpha"],data_dalpha,max_dalpha,bins_dalpha,0.5)
        #--degrees
        mk = max(data["data_k"])+5
        if mk > max_k:
            #extension de la distribution
            max_k = mk
            l1 = len(bins_k)-1
            bins_k = range(0,max_k,5)
            l2 = len(bins_k)-1
            fill = [0 for n in data_k[0]]
            for i in range(l1,l2):
                data_k.append(fill[:])
        dist = np.histogram(data["data_k"],bins=bins_k,density=True)
        for i,n in enumerate(dist[0]):
            data_k[i].append(n)
        #--strengths
        max_s,bins_s = distrib_fill(data["data_s"],data_s,max_s,bins_s,0.25)
        #--weights
        max_w,bins_w = distrib_fill(data["data_w"],data_w,max_w,bins_w,0.5)
        #--number of contacts
        max_n,bins_n = distrib_fill(data["data_n"],data_n,max_n,bins_n,0.5)
        #--contact durations
        max_tau,bins_tau = distrib_fill(data["data_tau"],data_tau,max_tau,bins_tau,0.5)
        #--intercontact durations
        max_dtau,bins_dtau = distrib_fill(data["data_dtau"],data_dtau,max_dtau,bins_dtau,0.5)

        #analyse
        conv_TL.append(distrib_analysis(data_TL,bins_TL,data_init["data_TL"]["data"],bins_TL))
        conv_a.append(distrib_analysis(data_a,bins_a,data_init["data_a"]["data"],data_init["data_a"]["bins"]))
        conv_alpha.append(distrib_analysis(data_alpha,bins_alpha,data_init["data_alpha"]["data"],data_init["data_alpha"]["bins"]))
        conv_dalpha.append(distrib_analysis(data_dalpha,bins_dalpha,data_init["data_dalpha"]["data"],data_init["data_dalpha"]["bins"]))
        conv_k.append(distrib_analysis(data_k,bins_k,data_init["data_k"]["data"],data_init["data_k"]["bins"]))
        conv_s.append(distrib_analysis(data_s,bins_s,data_init["data_s"]["data"],data_init["data_s"]["bins"]))
        conv_w.append(distrib_analysis(data_w,bins_w,data_init["data_w"]["data"],data_init["data_w"]["bins"]))
        conv_n.append(distrib_analysis(data_n,bins_n,data_init["data_n"]["data"],data_init["data_n"]["bins"]))
        conv_tau.append(distrib_analysis(data_tau,bins_tau,data_init["data_tau"]["data"],data_init["data_tau"]["bins"]))
        conv_dtau.append(distrib_analysis(data_dtau,bins_dtau,data_init["data_dtau"]["data"],data_init["data_dtau"]["bins"]))

    sortie = open("conv_"+m+".dat",'w')
    output = {"data_TL":conv_TL,
              "data_a":conv_a,
              "data_alpha":conv_alpha,
              "data_dalpha":conv_dalpha,
              "data_k":conv_k,
              "data_s":conv_s,
              "data_w":conv_w,
              "data_n":conv_n,
              "data_tau":conv_tau,
              "data_dtau":conv_dtau}
    dump(output,sortie)
    sortie.close()

#------------------------------------------

list_m = ["n_pTheta","L_pTheta","pTheta_k","pTheta","pttau","pitau_pidtau_t1","pitau_pidtau","pitau_t1_tw","pitau","L_ptau"]

#donnees de reference
entree = open("analysis_init.dat")
data_init = load(entree)
entree.close()
#--activity timeline
data_TL = data_init["data_TL"]
#--activity
max_a = max(data_init["data_a"])+0.5
bins_init_a = 2.**(np.arange(0,np.log2(max_a),0.5))
dist = np.histogram(data_init["data_a"],bins=bins_init_a,density=True)
data_a = dist[0][:]
#--activity durations
max_alpha = max(data_init["data_alpha"])+0.5
bins_init_alpha = 2.**(np.arange(0,np.log2(max_alpha),0.5))
dist = np.histogram(data_init["data_alpha"],bins=bins_init_alpha,density=True)
data_alpha = dist[0][:]
#--inactivity durations
max_dalpha = max(data_init["data_dalpha"])+0.5
bins_init_dalpha = 2.**(np.arange(0,np.log2(max_dalpha),0.5))
dist = np.histogram(data_init["data_dalpha"],bins=bins_init_dalpha,density=True)
data_dalpha = dist[0][:]
#--degrees
max_k = max(data_init["data_k"])+5
bins_init_k = range(0,max_k,5)
dist = np.histogram(data_init["data_k"],bins=bins_init_k,density=True)
data_k = dist[0][:]
#--strengths
max_s = max(data_init["data_s"])+0.5
bins_init_s = 2.**(np.arange(0,np.log2(max_s),0.5))
dist = np.histogram(data_init["data_s"],bins=bins_init_s,density=True)
data_s = dist[0][:]
#--weights
max_w = max(data_init["data_w"])+0.5
bins_init_w = 2.**(np.arange(0,np.log2(max_w),0.5))
dist = np.histogram(data_init["data_w"],bins=bins_init_w,density=True)
data_w = dist[0][:]
#--number of contacts
max_n = max(data_init["data_n"])+0.5
bins_init_n = 2.**(np.arange(0,np.log2(max_n),0.5))
dist = np.histogram(data_init["data_n"],bins=bins_init_n,density=True)
data_n = dist[0][:]
#--contact durations
max_tau = max(data_init["data_tau"])+0.5
bins_init_tau = 2.**(np.arange(0,np.log2(max_tau),0.5))
dist = np.histogram(data_init["data_tau"],bins=bins_init_tau,density=True)
data_tau = dist[0][:]
#--intercontact durations
max_dtau = max(data_init["data_dtau"])+0.5
bins_init_dtau = 2.**(np.arange(0,np.log2(max_dtau),0.5))
dist = np.histogram(data_init["data_dtau"],bins=bins_init_dtau,density=True)
data_dtau = dist[0][:]

data_init = {"data_TL":{"data":data_TL},
             "data_a":{"data":data_a,"bins":bins_init_a},
             "data_alpha":{"data":data_alpha,"bins":bins_init_alpha},
             "data_dalpha":{"data":data_dalpha,"bins":bins_init_dalpha},
             "data_k":{"data":data_k,"bins":bins_init_k},
             "data_s":{"data":data_s,"bins":bins_init_s},
             "data_w":{"data":data_w,"bins":bins_init_w},
             "data_n":{"data":data_n,"bins":bins_init_n},
             "data_tau":{"data":data_tau,"bins":bins_init_tau},
             "data_dtau":{"data":data_dtau,"bins":bins_init_dtau}}

for m in list_m:
    print m
    med_std(m,data_init)
