#computes the median and standard deviation of the measures, for a list of randomisations

from glob import glob
from pickle import load,dump
import numpy as np

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

def distrib_analysis(data_n):
    med_n,inf_n,sup_n = [],[],[]
    for n in data_n:
        med_n.append(np.median(n))
        inf_n.append(np.percentile(n,5))
        sup_n.append(np.percentile(n,95))
    return med_n,inf_n,sup_n

def med_std(m):
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
    print "Init OK"

    list_f = list_f[:10]
    
    #remplissage
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
    print "Fill OK"

    #analyse
    print "Analysis..."
    #--activity timeline
    med_TL,inf_TL,sup_TL = distrib_analysis(data_TL)
    #--activity
    med_a,inf_a,sup_a = distrib_analysis(data_a)
    #--activity durations
    med_alpha,inf_alpha,sup_alpha = distrib_analysis(data_alpha)
    #--inactivity durations
    med_dalpha,inf_dalpha,sup_dalpha = distrib_analysis(data_dalpha)
    #--degrees
    med_k,inf_k,sup_k = distrib_analysis(data_k)
    #--strengths
    med_s,inf_s,sup_s = distrib_analysis(data_s)
    #--weights
    med_w,inf_w,sup_w = distrib_analysis(data_w)
    #--number of contacts
    med_n,inf_n,sup_n = distrib_analysis(data_n)
    #--contact durations
    med_tau,inf_tau,sup_tau = distrib_analysis(data_tau)
    #--intercontact durations
    med_dtau,inf_dtau,sup_dtau = distrib_analysis(data_dtau)
    print "Analysis OK"

    sortie = open("med_"+m+".dat",'w')
    output = {"data_TL":{"med":med_TL,"inf":inf_TL,"sup":sup_TL},
              "data_a":{"med":med_a,"inf":inf_a,"sup":sup_a,"bins":bins_a},
              "data_alpha":{"med":med_alpha,"inf":inf_alpha,"sup":sup_alpha,"bins":bins_alpha},
              "data_dalpha":{"med":med_dalpha,"inf":inf_dalpha,"sup":sup_dalpha,"bins":bins_dalpha},
              "data_k":{"med":med_k,"inf":inf_k,"sup":sup_k,"bins":bins_k},
              "data_s":{"med":med_s,"inf":inf_s,"sup":sup_s,"bins":bins_s},
              "data_w":{"med":med_w,"inf":inf_w,"sup":sup_w,"bins":bins_w},
              "data_n":{"med":med_n,"inf":inf_n,"sup":sup_n,"bins":bins_n},
              "data_tau":{"med":med_tau,"inf":inf_tau,"sup":sup_tau,"bins":bins_tau},
              "data_dtau":{"med":med_dtau,"inf":inf_dtau,"sup":sup_dtau,"bins":bins_dtau}}
    dump(output,sortie)
    sortie.close()

#------------------------------------------

list_m = ["n_pTheta","L_pTheta","pTheta_k","pTheta","pttau","pitau_pidtau_t1","pitau_pidtau","pitau_t1_tw","pitau","L_ptau"]

for m in list_m:
    print m
    med_std(m)
