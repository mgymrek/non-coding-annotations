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
import pyfasta

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
        
def plot_context_count(contextinfo, outprefix, smooth, key):
    """
    Plot composite plot of:
    1. context count
    2. variant count
    3. expected count
    """
    xcoord = sorted(contextinfo.keys())[1:-1]
    ycoord1 = [contextinfo[c].get(key, [1, 1, 1])[0] for c in xcoord]
    ycoord2 = [contextinfo[c].get(key, [1, 1, 1])[1] for c in xcoord]
    ycoord3 = [ycoord2[i]*1.0/ycoord1[i] for i in range(len(ycoord1))]
    color = "blue"
    if IsCpG(key): color="green"
    fig = plt.figure()
    fig.set_size_inches((10, 5))
    ax = fig.add_subplot(311)
    ax.scatter(xcoord, ycoord1, color=color, alpha=0.2)
    ax.plot(xcoord, Smooth(ycoord1, smooth), color="red")
    ax.set_ylabel("Context count")
    ax.set_title(key)
    ax = fig.add_subplot(312)
    ax.scatter(xcoord, ycoord2, color=color, alpha=0.2)
    ax.plot(xcoord, Smooth(ycoord2, smooth), color="red")
    ax.set_ylabel("Variant count")
    ax = fig.add_subplot(313)
    ax.scatter(xcoord, ycoord3, color=color, alpha=0.2)
    y1s = Smooth(ycoord1, smooth)
    y2s = Smooth(ycoord2, smooth)
    y3s = [y2s[i]*1.0/y1s[i] for i in range(len(y1s))]
    ax.plot(xcoord, y3s, color="red")
    ax.axhline(np.mean(ycoord3), linestyle="dashed", color="black", lw=2)
    ax.set_ylabel("Fraction")
    ax.set_ylim(bottom=min(ycoord3), top=max(ycoord3))
    fig.tight_layout()
    fig.savefig("%s_%s_contextplot.png"%(outprefix, key))
    plt.close()

def GetObserved(info, cpg=True):
    obs = 0
    keys = [item for item in info.keys() if item != "all"]
    if not cpg:
        keys = [item for item in keys if not IsCpG(item)]
    for k in keys:
        obs += info[k][1]
    return obs

def GetExpected(info, mutrates, cpg=True):
    exp = 0
    keys = [item for item in info.keys() if item != "all"]
    if not cpg:
        keys = [item for item in keys if not IsCpG(item)]
    for k in keys:
        mutkeys = [item for item in mutrates.keys() if item.split(":")[0] == k]
        m = sum([mutrates[mk] for mk in mutkeys])
        count = info[k][0]
        exp += count*m
    return exp

def plot_exp_count(contextinfo, mutrates, outprefix, smooth):
    xcoord = sorted(contextinfo.keys())[1:-1]
    yobs = [contextinfo[c].get("all", [1, 1, 1])[1] for c in xcoord]
    yexp = [GetExpected(contextinfo[c], mutrates) for c in xcoord]
    yobs_nocpg = [GetObserved(contextinfo[c], cpg=False) for c in xcoord]
    yexp_nocpg = [GetExpected(contextinfo[c], mutrates, cpg=False) for c in xcoord]
    fig = plt.figure()
    fig.set_size_inches((10, 10))
    ax = fig.add_subplot(611)
    ax.scatter(xcoord, yobs, color="blue")
    ax.plot(xcoord, Smooth(yobs, smooth), color="red")
    ax.set_ylabel("Observed")
    ax = fig.add_subplot(612)
    ax.scatter(xcoord, yexp, color="blue")
    ax.set_ylim(bottom=min(yexp), top=max(yexp))
    ax.plot(xcoord, Smooth(yexp, smooth), color="red")
    ax.set_ylabel("Expected")
    ax = fig.add_subplot(613)
    yfrac = np.log([yobs[i]/yexp[i]*np.mean(yexp)/np.mean(yobs) for i in range(len(yobs))])/np.log(2)
    ax.scatter(xcoord, yfrac, color="blue")
    ax.plot(xcoord, Smooth(yfrac, smooth), color="red")
    ax.axhline(0)
    ax.set_ylabel("Ratio")
    ax = fig.add_subplot(614)
    ax.scatter(xcoord, yobs_nocpg, color="blue")
    ax.set_ylim(bottom=min(yobs_nocpg), top=max(yobs_nocpg))
    ax.plot(xcoord, Smooth(yobs_nocpg, smooth), color="red")
    ax.set_ylabel("Obs - no cpg")
    ax = fig.add_subplot(615)
    ax.scatter(xcoord, yexp_nocpg, color="blue")
    ax.set_ylim(bottom=min(yexp_nocpg), top=max(yexp))
    ax.plot(xcoord, Smooth(yexp_nocpg, smooth), color="red")
    ax.set_ylim(bottom=min(yexp_nocpg), top=max(yexp_nocpg))
    ax.set_ylabel("Exp - no cpg")
    ax = fig.add_subplot(616)
    yfrac_nocpg = np.log([yobs_nocpg[i]/yexp_nocpg[i]*np.mean(yexp_nocpg)/np.mean(yobs_nocpg) for i in range(len(yobs_nocpg))])/np.log(2)
    ax.scatter(xcoord, yfrac_nocpg, color="blue")
    ax.plot(xcoord, Smooth(yfrac_nocpg, smooth), color="red")
    ax.axhline(0)
    ax.set_ylabel("Ratio")
    fig.tight_layout()
    fig.savefig("%s_obsexp.png"%outprefix)
    plt.close()

def plot_sing_count(locinfo, outprefix, smooth, key):
    """
    Plot composite plot of % singletons and variant count
    """
    xcoord = sorted(locinfo.keys())[1:-1]
    ycoord = [locinfo[c].get(key, [1, 1])[1]*1.0/sum(locinfo[c].get(key, [1, 1, 1])[0:2]) for c in xcoord]
    ycoord2 = [sum(locinfo[c].get(key, [1, 1, 1])[0:2]) for c in xcoord]
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
    ycoord = [sum(locinfo[c].get(key, [1, 1, 1])[0:2]) for c in xcoord]
    ycoord2 = [sum(locinfo[c].get(rkey, [1, 1, 1])[0:2]) for c in xcoord]
    fig = plt.figure()
    fig.set_size_inches((10, 5))
    ax = fig.add_subplot(211)
    color="blue"
    if IsCpG(key.split(":")[0]): color="green"
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

def IsCpG(context):
    # CGX rev is XCG
    # XCG
    return "CG" in context

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--variants", help="File with columns chrom, start, rsid, ref, alt, qual, filter, acount, afreq", required=True, type=str)
    parser.add_argument("--context", help="Break down by context. variant columns are instead chrom, start, rsid, ref, alt, maf, count, ancestral, fromcontext, tocontext", action="store_true")
    parser.add_argument("--bed", help="File of chrom, start, end, strand of variants to analyze", required=True, type=str)
    parser.add_argument("--out", help="Output prefix", required=True, type=str)
    parser.add_argument("--window", help="Look this many bp up and downstream", required=False, default=1000, type=int)
    parser.add_argument("--smooth", help="Smooth values param", default=5, type=int)
    parser.add_argument("--remove-cpg", help="Remove CpG sites from variant count plots", action="store_true")
    parser.add_argument("--mutrates", help="Mutation rate table", type=str, default=None)
    parser.add_argument("--ref", help="Genome reference fasta", type=str, default="/humgen/atgu1/fs03/wip/gymrek/genomes/Homo_sapiens_assembly19.fasta")
    args = parser.parse_args()

    mutrates = {}
    if args.mutrates:
        with open(args.mutrates, "r") as f:
            for line in f:
                if "from" in line: continue
                items = line.strip().split()
                mutrates["%s:%s"%(items[0], items[1])] = float(items[2])
                         
    genome = pyfasta.Fasta(args.ref)
    # get keys
    chromToKey = {}
    for k in genome.keys():
        chrom = k.split()[0]
        chromToKey[chrom] = k

    locinfo = {} # map pos -> context-> (nonsingcount, singcount)
    coords = range(-1*args.window, args.window+1)
    for c in coords:
        locinfo[c] = {}
        locinfo[c]["all"] = [0, 0]
    contextinfo = {} # map pos->fromcontext-> (context_count, context_variant_observed, context_variant_expected)
    for c in coords:
        contextinfo[c] = {}
        contextinfo[c]["all"] = [0, 0, 0]

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

            # Get background context
            if args.context:
                for rstart in range(start-args.window, start+args.window):
                    ref_context = genome[chromToKey[chrom]][(rstart-2):(rstart+1)]
                    if strand == "-":
                        ref_context = ReverseComplement(ref_context)
                        coord = start-rstart
                    else: coord = rstart-start
#                    print chrom, start, end, strand, coord, ref_context
                    contextinfo[coord]["all"][0] += 1
                    if ref_context not in contextinfo[coord]: contextinfo[coord][ref_context] = [0,0,0]
                    contextinfo[coord][ref_context][0] += 1
#                    print coord, contextinfo[coord]

            # Get variants
            for record in windowvars:
                if record[6] != "PASS" and not args.context: continue
                rstart = int(record[1])
                acount = int(record[7])
                is_sing = (acount==1)
                if strand == "+":
                    pos = rstart-start
                else: pos = start-rstart
                if args.context:
                    ref = record[4]
                    ancestral = record[8]
                    fromcontext = record[9]
                    tocontext = record[10]
                    if strand == "-":
                        fromcontext = ReverseComplement(fromcontext)
                        tocontext = ReverseComplement(tocontext)
                    if IsCpG(fromcontext) and args.remove_cpg: continue
                    context = "%s:%s"%(fromcontext, tocontext)
                    if context not in locinfo[pos].keys(): locinfo[pos][context] = [0, 0]
                    locinfo[pos][context][is_sing] = locinfo[pos][context][is_sing]+1
                    if fromcontext not in contextinfo[pos]: contextinfo[pos][fromcontext] = [0,0,0]
#                    print record, tocontext, pos, contextinfo[pos]
                    if ancestral != ref: # Correct for if ancestral is not ref
                        contextinfo[pos][tocontext][0] -= 1
                        contextinfo[pos][fromcontext][0] += 1
                    contextinfo[pos][fromcontext][1] += 1
                    contextinfo[pos]["all"][1] += 1
                locinfo[pos]["all"][is_sing] = locinfo[pos]["all"][is_sing]+1
            count = count + 1                       
            print count
            if count % 10000 == 0: sys.stderr.write("Processed %s loci\n"%count)

    # Add pseoducounts
    for c in coords:
        for key in locinfo[c]:
            locinfo[c][key][0] += 1
            locinfo[c][key][1] += 1
        for key in contextinfo[c]:
            contextinfo[c][key][0] += 1
            contextinfo[c][key][1] += 1

    # Summarize
    locfile = open("%s_locinfo.tab"%args.out, "w")
    keys = set()
    for c in coords:
        for key in locinfo[c]:
            keys.add(key)
            locfile.write("\t".join(map(str, [c, key, locinfo[c][key][0], locinfo[c][key][1], sum(locinfo[c][key][0:2]), locinfo[c][key][1]*1.0/sum(locinfo[c][key][0:2])]))+"\n")
    locfile.close()
    contextfile = open("%s_contextinfo.tab"%args.out, "w")
    fromkeys = set()
    for c in coords:
        for key in contextinfo[c]:
            fromkeys.add(key)
            contextfile.write("\t".join(map(str, [c, key, contextinfo[c][key][0], contextinfo[c][key][1], contextinfo[c][key][1]/contextinfo[c][key][0]]))+"\n")
    contextfile.close()

    # Plot
#    for key in keys:
#        plot_sing_count2(locinfo, args.out, args.smooth, key)
#        plot_sing_count(locinfo, args.out, args.smooth, key)
    plot_exp_count(contextinfo, mutrates, args.out, args.smooth)
    for key in fromkeys:
        plot_context_count(contextinfo, args.out, args.smooth, key)

main()
    
