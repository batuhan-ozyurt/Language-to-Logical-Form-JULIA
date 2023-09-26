# Language-to-Logical-Form-JULIA

Paper: [Language to Logical Form with Neural Attention by Li Dong, Mirella Lapata](https://arxiv.org/abs/1601.01280))

This is the Julia implementation of the paper above.

It implements the four methods mentioned in the paper: SEQ2SEQ, SEQ2SEQ with Attention, SEQ2TREE, SEQ2TREE with Attention.

To run the code, open the notebook files and write the path to the training and test data.

Dropout value should be set to 0.4 for the GEO and JOBS datasets and 03. for the ATIS dataset.

Datasets can be downloaded via the following links:

SEQ2SEQ

JOBS: http://dong.li/lang2logic/seq2seq_jobqueries.zip

GEO: http://dong.li/lang2logic/seq2seq_geoqueries.zip

ATIS: http://dong.li/lang2logic/seq2seq_atis.zip

SEQ2TREE

JOBS: http://dong.li/lang2logic/seq2tree_jobqueries.zip

GEO: http://dong.li/lang2logic/seq2tree_geoqueries.zip

ATIS: http://dong.li/lang2logic/seq2tree_atis.zip
