-----------------------------------------------------------------------------
pr-graph version 0.1 (Graph-based Posterior Regularization
-----------------------------------------------------------------------------

----------------
Contents
----------------

1. Compiling
2. Graph Building
   a. POS Tagging Graph
   b. Handwriting Letters Graph
3. Running
   a. Input data format
   b. Running PR-graph
   c. Reproducing results in CoNLL-2013

----------------
1. Compiling
----------------

The build.xml is included in the repository.
Use Ant (http://ant.apache.org/) to compile the project.

---------------------------------
2. Graph Buiding
--------------------------------

---------------------------------
2a. POS Tagging Graph
--------------------------------
PosTagging Graph building uses a suffix dictionary included in //pr-graph/data/suffix.dict
Universal part-of-speech tags mapping can be found here:
https://code.google.com/p/universal-pos-tags/

To run the graph builder, we can do:

export WDIR="your working directory"
export DDIR=”your data directory”
export CLASSPATH="$WDIR/bin/:$WDIR/libs/optimization-2010.11.jar:$WDIR/libs/trove-2.0.2.jar:$WDIR/libs/args4j-2.0.10.jar"

java -cp $CLASSPATH -Xmx8000m programs.TestPosGraphBuilder  \
-data-path "DDIR/lang.train,$DDIR/lang.test" \ # a list of comma-delimited input file paths
-sufix-path “DDIR/suffix.dict”
-umap-path "$DDIR/lang.map" \
-graph-path "$DDIR/graph/lang.grph" \ 
-ngram-path "$DDIR/graph/$lang.idx" \ 
-num-neighbors 60 \
-lang-name "lang"

The Graph builder outputs the node index file to -ngram-path, and the graph edge file to -graph-path. More options can be found at config.Config, config.PosConfig and config.PosGraphConfig.

-----------------------
2b. Handwriting Letters Graph
-----------------------
The code for building OCR Graph lives in another project (due to its dependency on FastEMD code) and will be uploaded soon. Right now we can use the graph file in //data/graph to run the experiments.

-------------------------
3. Running
-------------------------

-------------------------
3a. Input data format
-------------------------
We use the CoNLL-X (http://ilk.uvt.nl/conll/index.html#dataformat) format for POSTagging, and the OCR (http://www.seas.upenn.edu/~taskar/ocr/) data for the handwriting task.

----------------------------
3b. Running PR-graph
----------------------------



