#!/usr/bin/env python

"""
Generate composite plot of local allele frequencies
centered around bed coordinates
"""

import argparse
import gzip
import sys
import tabix

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--variants", help="File with columns chrom, start, rsid, ref, alt, qual, filter, acount, afreq", required=True, type=str)
    parser.add_argument("--bed", help="File of chrom, start, end, strand of variants to analyze", required=True, type=str)
    parser.add_argument("--out", help="Output prefix", required=True, type=str)
    parser.add_argument("--window", help="Look this many bp up and downstream", required=False, default=1000, type=int)
    args = parser.parse_args()

    locinfo = {} # map pos -> (totalcount, singcount)
    coords = range(-1*args.window, args.window+1)
    for c in coords: locinfo[c] = [1, 1] # pseudocount to each

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
                rstart = int(record[1])
                acount = int(record[7])
                is_sing = (acount==1)
                if strand == "+":
                    pos = rstart-start
                else: pos = start-rstart
                locinfo[pos][is_sing] = locinfo[pos][is_sing]+1
            count = count + 1
            if count % 10000 == 0: sys.stderr.write("Processed %s loci\n"%count)
    
    # Summarize
    locfile = open("%s_locinfo.tab"%args.out, "w")
    for c in coords:
        locfile.write("\t".join(map(str, [c, locinfo[c][0], locinfo[c][1], sum(locinfo[c]), locinfo[c][1]*1.0/sum(locinfo[c])]))+"\n")
    locfile.close()

main()
    
