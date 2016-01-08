#!/usr/bin/env python

import h5py
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
    parser.add_option('-i', dest='index_snp', default=False, action='store_true', help='SNPs are labeled with their index SNP as column 6 [Default: %default]')
    parser.add_option('-s', dest='score', default=False, action='store_true', help='SNPs are labeled with scores as column 7 [Default: %default]')
    parser.add_option('-o', dest='out_dir', default='sad', help='Output directory for tables and plots [Default: %default]')
    parser.add_option('-l', dest='seq_len', type='int', default=600, help='Sequence length provided to the model [Default: %default]')

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

    # Prepare hdf5 file
    h5f = h5py.File('%s/model_in.h5'%options.out_dir, 'w')
    dset = h5f.create_dataset('test_in', (0, 4, 1, options.seq_len), maxshape=(None, 4, 1, options.seq_len), dtype='i8')

    # Read through VCF
    with open(vcf_file, "r") as f:
        for line in f:
            # Get one hot coded sequence
            snp = vcf.SNP(line, index_snp=options.index_snp, score=options.score)
            seq_vecs, seqs, seq_headers = vcf.snps_seq1([snp], options.genome_fasta, options.seq_len)
            seq_vecs = seq_vecs.reshape((seq_vecs.shape[0],4,1,seq_vecs.shape[1]/4))
            # Add to hdf5 file
            current_shape = dset.shape[0]
            dset.resize(current_shape+seq_vecs.shape[0], axis=0)
            dset[current_shape:,...] = seq_vecs
    h5f.close()

    #################################################################
    # predict in Torch
    #################################################################
    model_hdf5_file = '%s/model_out.txt' % options.out_dir
    cuda_str = ""
    cmd = 'basset_predict.lua -norm %s %s %s/model_in.h5 %s' % (cuda_str, model_th, options.out_dir, model_hdf5_file)
    if subprocess.call(cmd, shell=True):
        message('Error running basset_predict.lua', 'error')

################################################################################
# __main__
################################################################################
if __name__ == '__main__':
    main()
