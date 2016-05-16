#!/usr/bin/env python

"""
Generate composite plot of local allele frequencies
centered around bed coordinates
"""

import matplotlib as mpl
mpl.use('Agg')

import argparse
import gzip
import math
import matplotlib.pyplot as plt
import numpy as np
import sys
import tabix

def ReverseComplement(nucs):
    newnucs = ""
    for i in range(len(nucs)):
        if nucs[i] == "A": newnucs += "T"
        elif nucs[i] == "T": newnucs += "A"
        elif nucs[i] == "C": newnucs += "G"
        elif nucs[i] == "G": newnucs += "C"
        else: newnucs += nucs[i] # Takes care of ":" in keys
    return newnucs[::-1]

def Smooth(data, num):
    """
    each point is mean of +/- num points
    """
    newdata = []
    for i in range(len(data)):
        lb = max([i-num, 0])
        ub = min([len(data)-1, i+num])
        newdata.append(np.mean(data[lb:ub]))
    return newdata
        

def plot_sing_count(locinfo, outprefix, smooth, key):
    """
    Plot composite plot of % singletons and variant count
    """
    xcoord = sorted(locinfo.keys())[1:-1]
    ycoord = [locinfo[c].get(key, [1, 1])[1]*1.0/sum(locinfo[c].get(key, [1, 1])) for c in xcoord]
    ycoord2 = [sum(locinfo[c].get(key, [1, 1])) for c in xcoord]
    fig = plt.figure()
    fig.set_size_inches((10, 5))
    ax = fig.add_subplot(211)
    ax.scatter(xcoord, ycoord, color="blue", alpha=0.2)
    ax.plot(xcoord, Smooth(ycoord, smooth), color="red")
    ax.set_ylabel("% Singleton")
    ax = fig.add_subplot(212)
    ax.scatter(xcoord, map(lambda x: math.log10(x), ycoord2), color="blue", alpha=0.2)
    ax.plot(xcoord, Smooth(map(lambda x: math.log10(x), ycoord2), smooth), color="red")
    ax.set_ylabel("Variant count")
    fig.tight_layout()
    fig.savefig("%s_%s_plot.png"%(outprefix, key))
    plt.close()

def plot_sing_count2(locinfo, outprefix, smooth, key):
    """
    Plot composite plot of % singletons and variant count
    """
    print " ### Plotting %s ###"%key
    if key == "all": rkey = "all"
    else: rkey = "%s:%s"%(ReverseComplement(key.split(":")[0]), ReverseComplement(key.split(":")[1]))
    print key, rkey
    xcoord = sorted(locinfo.keys())[1:-1]
    ycoord = [sum(locinfo[c].get(key, [1, 1])) for c in xcoord]
    ycoord2 = [sum(locinfo[c].get(rkey, [1, 1])) for c in xcoord]
    fig = plt.figure()
    fig.set_size_inches((10, 5))
    ax = fig.add_subplot(211)
    color="blue"
    if "CG" in key.split(":")[0]: color="green" 
    ax.scatter(xcoord, map(lambda x: math.log10(x), ycoord), color=color, alpha=0.2)
    ax.plot(xcoord, Smooth(map(lambda x: math.log10(x), ycoord), smooth), color="red")
    ax = fig.add_subplot(212)
    ax.scatter(xcoord, map(lambda x: math.log10(x), ycoord2), color=color, alpha=0.2)
    ax.plot(xcoord, Smooth(map(lambda x: math.log10(x), ycoord2), smooth), color="red")
    ax.set_ylabel("Variant count")
    ax.set_title("Top=%s; Bottom=%s"%(key, rkey))
    fig.tight_layout()
    fig.savefig("%s_%s_plot_v2.png"%(outprefix, key))
    plt.close()

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--variants", help="File with columns chrom, start, rsid, ref, alt, qual, filter, acount, afreq", required=True, type=str)
    parser.add_argument("--context", help="Break down by context. variant columns are instead chrom, start, rsid, ref, alt, maf, count, ancestral, fromcontext, tocontext", action="store_true")
    parser.add_argument("--bed", help="File of chrom, start, end, strand of variants to analyze", required=True, type=str)
    parser.add_argument("--out", help="Output prefix", required=True, type=str)
    parser.add_argument("--window", help="Look this many bp up and downstream", required=False, default=1000, type=int)
    parser.add_argument("--smooth", help="Smooth values param", default=5, type=int)
    args = parser.parse_args()

    locinfo = {} # map pos -> context-> (nonsingcount, singcount)
    coords = range(-1*args.window, args.window+1)
    for c in coords:
        locinfo[c] = {}
        locinfo[c]["all"] = [0, 0] # pseudocount to each

    tb = tabix.open(args.variants)
    count = 0
    with gzip.open(args.bed, "r") as bedfile:
        for line in bedfile:
            chrom, start, end, strand = line.strip().split()[0:4]
            start = int(start)
            end = int(end)
            if start-args.window < 0: continue
            try:
                windowvars = tb.query(chrom, start-args.window, end+args.window-1)
            except tabix.TabixError: continue
            for record in windowvars:
                if record[6] != "PASS" and not args.context: continue
                rstart = int(record[1])
                acount = int(record[7])
                is_sing = (acount==1)
                if strand == "+":
                    pos = rstart-start
                else: pos = start-rstart
                locinfo[pos]["all"][is_sing] = locinfo[pos]["all"][is_sing]+1
                if args.context:
                    fromcontext = record[9]
                    tocontext = record[10]
                    if strand == "-":
                        fromcontext = ReverseComplement(fromcontext)
                        tocontext = ReverseComplement(tocontext)
                    context = "%s:%s"%(fromcontext, tocontext)
                    if context not in locinfo[pos].keys(): locinfo[pos][context] = [0, 0]
                    locinfo[pos][context][is_sing] = locinfo[pos][context][is_sing]+1
            count = count + 1
            if count % 10000 == 0: sys.stderr.write("Processed %s loci\n"%count)
    
    # Add pseoducount
    for c in coords:
        for key in locinfo[c]:
            locinfo[c][key][0] += 1
            locinfo[c][key][1] += 1

    # Summarize
    locfile = open("%s_locinfo.tab"%args.out, "w")
    keys = set()
    for c in coords:
        for key in locinfo[c]:
            keys.add(key)
            locfile.write("\t".join(map(str, [c, key, locinfo[c][key][0], locinfo[c][key][1], sum(locinfo[c][key]), locinfo[c][key][1]*1.0/sum(locinfo[c][key])]))+"\n")
    locfile.close()

    # Plot
    for key in keys:
        plot_sing_count2(locinfo, args.out, args.smooth, key)
        plot_sing_count(locinfo, args.out, args.smooth, key)

main()
    
