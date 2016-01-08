#!/usr/bin/env python

import h5py
import numpy as np
from optparse import OptionParser
import os
import subprocess
import sys

# Load Basset functions
sys.path = [os.path.join(os.environ["BASSETDIR"], "src")] + sys.path
import vcf
from util.utils import message

def main():
    usage = "usage: %prog [options] <model_th> <vcf_file>"
    parser = OptionParser(usage)
    parser.add_option('-f', dest='genome_fasta', default='%s/data/genomes/hg19.fa'%os.environ['BASSETDIR'], help='Genome FASTA from which sequences will be drawn [Default: %default]')
    parser.add_option('-b', dest='batchsize', default=128, help='Batch size for prediction. [Default: %default]')
    parser.add_option('-i', dest='index_snp', default=False, action='store_true', help='SNPs are labeled with their index SNP as column 6 [Default: %default]')
    parser.add_option('-s', dest='score', default=False, action='store_true', help='SNPs are labeled with scores as column 7 [Default: %default]')
    parser.add_option('-o', dest='out_dir', default='sad', help='Output directory for tables and plots [Default: %default]')
    parser.add_option('-l', dest='seq_len', type='int', default=600, help='Sequence length provided to the model [Default: %default]')
    parser.add_option('-t', dest='targets_file', default=None, help='File specifying target indexes and labels in table format')

    (options,args) = parser.parse_args()
    
    if len(args) != 2:
        parser.error('Must provide Torch model and VCF file')
    else:
        model_th = args[0]
        vcf_file = args[1]

    if not os.path.isdir(options.out_dir):
        os.mkdir(options.out_dir)

    #################################################################
    # prep SNP sequences
    #################################################################
    message('prep SNP sequences')

    # Prepare hdf5 file
    h5f = h5py.File('%s/model_in.h5'%options.out_dir, 'w')
    dset = h5f.create_dataset('test_in', (1, 4, 1, options.seq_len), maxshape=(None, 4, 1, options.seq_len))

    # Read through VCF
    current_shape = 0 # start off with 0 (init to 1 bc got error otherwise)
    with open(vcf_file, "r") as f:
        for line in f:
            # Get one hot coded sequence
            snp = vcf.SNP(line, index_snp=options.index_snp, score=options.score)
            seq_vecs, seqs, seq_headers = vcf.snps_seq1([snp], options.genome_fasta, options.seq_len)
            seq_vecs = seq_vecs.reshape((seq_vecs.shape[0],4,1,seq_vecs.shape[1]/4))
            # Add to hd5 file
            dset.resize(current_shape+seq_vecs.shape[0], axis=0)
            dset[current_shape:,...] = seq_vecs
            current_shape = dset.shape[0]
    h5f.close()

    #################################################################
    # predict in Torch
    #################################################################
    message('predict in torch')

    model_hdf5_file = '%s/model_out.txt' % options.out_dir
    cuda_str = ""
    cmd = 'basset_predict.lua -batchsize %s -norm %s %s %s/model_in.h5 %s' % (options.batchsize, cuda_str, model_th, options.out_dir, model_hdf5_file)
    if subprocess.call(cmd, shell=True):
        message('Error running basset_predict.lua', 'error')


    #################################################################
    # collect and print SADs
    #################################################################
    message('collect and print SADs')

    if options.targets_file is not None:
        target_labels = [line.split()[0] for line in open(options.targets_file)]

    sad_out = open('%s/sad_scores_table.txt' % options.out_dir, 'w')
    header_cols = ['rsid', 'index', 'score', 'ref', 'alt'] + target_labels
    sad_out.write('\t'.join(header_cols)+'\n')

    # Read simultaneously from SNP and predictions file
    snp_reader = open(vcf_file, "r")
    snpline = snp_reader.readline().strip()
    while snpline.startswith("#"): snpline = snp_reader.readline().strip()
    pred_reader = open(model_hdf5_file, "r")
    predline = pred_reader.readline().strip()

    # Iterate through SNPs
    while snpline != "":
        snp = vcf.SNP(snpline, index_snp=options.index_snp, score=options.score)
        ref_pred = np.array([float(p) for p in predline.split()])
        predline = pred_reader.readline().strip()
        for alt_al in snp.alt_alleles:
            alt_pred = np.array([float(p) for p in predline.split()])
            predline = pred_reader.readline().strip()
        alt_sad = alt_pred - ref_pred # TODO assuming biallelic
        sad_out.write('\t'.join(map(str, [snp.rsid, snp.index_snp, snp.score, snp.ref_allele, snp.alt_alleles[0]] + \
                                   map(lambda x: '%7.4f'%x, list(alt_sad)))) + '\n')
        snpline = snp_reader.readline().strip()
    snp_reader.close()
    pred_reader.close()
    sad_out.close()

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
