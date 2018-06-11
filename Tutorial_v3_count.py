from pickle import load

entree = open("analysis_init.dat")
data_init = load(entree)
entree.close()

print len(data_init["data_tau"])
