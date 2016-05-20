#!/usr/bin/env python
"""
Generate frequency-based score for variant annotations
"""

import argparse
import gzip
import joblib
import numpy as np
import pyfasta
import scipy.stats
import sys
import tabix

def LoadSNPs(snpfile):
    snps = []
    if snpfile.endswith(".gz"):
        of = gzip.open
    else: of = open
    with of(snpfile, "r") as f:
        for line in f:
            chrom, start, end, strand = line.strip().split()[0:4]
            start = int(start)
            end = int(end)
            snps.append((chrom, start, end, strand))
    return snps

def GetCaseControlACCounts(snp, args):
    controlgen = ControlGenerator(snp, args.ref, args.vcf, args.restrict, \
                                      args.num_ctrls, args.window, args.match_context)
    case_ac = controlgen.GetCaseAC()
    ctrl_ac = controlgen.GetControlACs()
    if len(ctrl_ac) < args.num_ctrls:
        sys.stderr.write("Could not generate enough controls for %s\n"%str(snp))
        return None, None
    return case_ac, ctrl_ac

class ControlGenerator:
    def __init__(self, _snp, _ref, _vcf, _restrict, \
                     _num_ctrls, _window, _match_context):
        self.snp = _snp
        self.ref = pyfasta.Fasta(_ref)
        self.vcf = tabix.open(_vcf)
        if _restrict is not None:
            self.restrict = tabix.open(_restrict)
        else: self.restrict = None
        self.chromToKey = {}
        for k in self.ref.keys():
            chrom = k.split()[0]
            self.chromToKey[chrom] = k
        self.num_ctrls = _num_ctrls
        self.window = _window
        self.match_context = _match_context
        if self.match_context >= 0:
            self.snp_context = self.GetContext(self.snp)

    def GetSNPAC(self, snp):
        try:
            records = list(self.vcf.query(*snp[0:3]))
            if len(records)==0: return 0
            record = list(records)[0]
            ac = int(record[7])
            return ac
        except tabix.TabixError:
            return 0
        
    def GetCaseAC(self):
        return self.GetSNPAC(self.snp)

    def ReverseComplement(self, nucs):
        newnucs = ""
        for i in range(len(nucs)):
            if nucs[i] == "A": newnucs += "T"
            elif nucs[i] == "T": newnucs += "A"
            elif nucs[i] == "G": newnucs += "C"
            elif nucs[i] == "C": newnucs += "G"
            else: newnucs += "N"
        return newnucs[::-1]

    def GetContext(self, snp):
        chrom, start, end = snp[0:3]
        context = self.ref[self.chromToKey[chrom]][(start-self.match_context):(end+self.match_context)]
        rc = self.ReverseComplement(context)
        if rc < context: return rc
        else: return context

    def CheckContext(self, csnp):
        csnp_context = self.GetContext(csnp)
        return csnp_context == self.snp_context

    def IsRestricted(self, csnp):
        if self.restrict is None: return False
        try:
            records = list(self.restrict.query(*csnp))
            return len(records) > 0
        except tabix.TabixError: return False

    def GenerateMatchedControls(self):
        controls = []
        # Get potential SNP window
        chrom = self.snp[0]
        start = self.snp[1] - self.window
        end = self.snp[2] + self.window
        # Get eligible candidates
        for pos in range(start, end+1):
            csnp = (chrom, pos, pos+1)
            # Check if resricted
            if self.IsRestricted(csnp):
                continue
            # Check context
            if self.match_context == -1: # no requirement
                controls.append(csnp)
            else:
                if self.CheckContext(csnp):
                    controls.append(csnp)
        return controls

    def GetControlACs(self):
        # Generate all possible controls
        controls = self.GenerateMatchedControls()
        # Randomly choose
        if len(controls) < self.num_ctrls: return []
        controls_keep_ind = list(np.random.choice(range(len(controls)), size=self.num_ctrls, replace=False))
        controls_keep = [controls[i] for i in controls_keep_ind]
        # Get afreqs
        afreqs = [self.GetSNPAC(ck) for ck in controls_keep]
        return afreqs

def GetIndex(count):
    if count < 2: return count
    else: return 2

def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--debug", help="Print helpful debugging info", action="store_true")
    parser.add_argument("--numproc", help="Number of processors", type=int, default=1)
    parser.add_argument("--max-snps", help="Only process this many SNPs (for debugging)", type=int, default=np.inf)
    #############################################
    # Options for input files
    parser.add_argument("--ref", help="Reference fasta file", type=str, default="/humgen/atgu1/fs03/wip/gymrek/genomes/Homo_sapiens_assembly19.fasta")
    parser.add_argument("--vcf", help="Indexed and bgzipped VCF file, must have an AC field", type=str, required=True)
    parser.add_argument("--case-snps", help="Bed file with SNP positions with a given annotation", type=str, required=True)
    parser.add_argument("--restrict", help="Bed file with regions to be excluded for use as controls", type=str, required=False)

    # Options for parameters
    parser.add_argument("--num-ctrls", help="Number of controls to pick for each SNP", type=int, default=1000)
    parser.add_argument("--window", help="Window size to search for controls", type=int, default=1000000)
    parser.add_argument("--match-context", help="Match controls on this much context. -1=don't match, 0=match nuc, 1=trinuc context", type=int, default=0)

    # Options for output
    parser.add_argument("--out", help="Output prefix. Default stdout", type=str, required=False)
    #############################################
    args = parser.parse_args()

    # Get SNPs
    snps = LoadSNPs(args.case_snps)

    # Get VCF reader
    vcf = tabix.open(args.vcf)

    # Get controls for each SNP
    case_counts = [] # minor allele count
    ctrl_counts = [] # case snp -> count
    counts = joblib.Parallel(n_jobs=args.numproc, verbose=5)(joblib.delayed(GetCaseControlACCounts)(snps[i], args) for i in range(min([args.max_snps, len(snps)])))
    for i in range(len(counts)):
        if counts[0] is None: continue
        case_counts.append(counts[i][0])
        ctrl_counts.append(counts[i][1])

    # Print output
    case_counts_summary = np.array([0, 0, 0])
    ctrl_counts_summary = np.array([0, 0, 0])
    for i in range(len(case_counts)):
        x = np.array([0,0,0]) # case counts
        y = np.array([0,0,0]) # ctrl counts
        x[GetIndex(case_counts[i])] += 1
        for j in range(len(ctrl_counts[i])):
            y[GetIndex(ctrl_counts[i][j])] += 1
        case_counts_summary += x
        ctrl_counts_summary += y
    res = scipy.stats.chisquare(case_counts_summary, ctrl_counts_summary*sum(case_counts_summary)/sum(ctrl_counts_summary))
    if args.out is None:
        f = sys.stdout
    else: f = open(args.out, "w")
    f.write("\t".join(map(str, [sum(case_counts_summary), sum(ctrl_counts_summary), \
                                    ",".join(map(str, case_counts_summary)), ",".join(map(str, ctrl_counts_summary)), \
                                    res.statistic, res.pvalue]))+"\n")

if __name__ == "__main__":
    main()
