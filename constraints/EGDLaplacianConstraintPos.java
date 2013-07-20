package constraints;

import gnu.trove.TIntArrayList;

import java.util.Random;
import analysis.FirstOrderEGDMonitor;
import config.PosConfig;
import data.AbstractSequence;
import data.PosCorpus;
import data.PosSequence;
import data.SparseSimilarityGraph;

/*
public class EGDLaplacianConstraintPos  {
	double[][][][] edgeScore, oldEdgeScore, edgeMarginal;
	double[][][] nodeScore, oldNodeScore, nodeMarginal; 
	double[] oldLogNorm;
	
	int[][] node2factor;
	int[] unlabeled;
	int[][] sentenceBatches;
	int numUnlabeled;
	
	PosCorpus corpus;
	SparseSimilarityGraph graph;
	PosFeatureFunction featureFunction;
	PosConfig config;
	
	double lpStrength, klStrength;
	double backoff;
	double uniformInit;
	double eta0, eta;
	int S0;
	int numIterations;
	
	int numNodes, numSequences, numStates;
	
	int numThreads;
	Random sentencePicker;

	SentenceUpdateThread[] uthreads;
	SentenceMonitorThread[] mthreads;
	
	public double entropyObjective, likelihoodObjective, graphObjective;

	double stoppingCriteria = 1e-6;
	
	public EGDLaplacianConstraintPos(PosCorpus corpus, SparseSimilarityGraph graph, PosFeatureFunction featureFunction, PosConfig config)
	{
		this.corpus = corpus;
		this.graph = graph;
		this.featureFunction = featureFunction;
		this.config = config;
		
		this.numNodes = corpus.numNodes;
		this.numSequences = corpus.numInstances;
		this.numStates = corpus.numStates - 2;
		this.S0 = corpus.initialState;
		
		this.eta0 = config.initialLearningRate;
		this.eta = eta0;
		
		this.numThreads = config.numThreads;
		this.uniformInit = config.estepInit;
		this.backoff = config.backoff;
		this.lpStrength = config.graphRegularizationStrength;
		this.klStrength = 1.0 - lpStrength;
		
		this.numIterations = config.numEstepIters;
		
		System.out.println("Initializing EGD Contraint trainer ... ");
		System.out.println("Number of nodes:\t" + numNodes + "\tNumber of states\t" + numStates);
		
		edgeScore = new double[numSequences][][][];
		oldEdgeScore = new double[numSequences][][][];
		edgeMarginal = new double[numSequences][][][];
		
		nodeScore = new double[numSequences][][];
		oldNodeScore = new double[numSequences][][];
		nodeMarginal = new double[numSequences][][];
		
		oldLogNorm = new double[numSequences];
		node2factor = new int[numNodes][2];
		
		for(int i = 0; i < numSequences; i++) {
			PosSequence instance = (PosSequence) corpus.getInstance(i);
			edgeScore[i] = new double[instance.length][numStates][numStates+1];
			oldEdgeScore[i] = new double[instance.length][numStates][numStates+1];
			edgeMarginal[i] = new double[instance.length][numStates][numStates+1];
			
			nodeScore[i] = new double[instance.length][numStates];
			oldNodeScore[i] = new double[instance.length][numStates];
			nodeMarginal[i] = new double[instance.length][numStates];
			
			for(int j = 0; j < instance.length; j++) 
				if(instance.nodes[j] >= 0) {
					node2factor[instance.nodes[j]][0] = i;
					node2factor[instance.nodes[j]][1] = j;
				}
		}
		
		unlabeled = corpus.unlabeled;
		numUnlabeled = unlabeled.length;
		sentencePicker = new Random(12345);
		
		uthreads = new SentenceUpdateThread[numThreads];
		mthreads = new SentenceMonitorThread[numThreads];
		sentenceBatches = new int[numThreads][];
		
		for(int i = 0; i < numThreads; i++) {
			int bsize = unlabeled.length / numThreads;
			TIntArrayList sids = new TIntArrayList();
			for(int j = i * bsize; j < (i + 1 < numThreads ? (i + 1) * bsize : unlabeled.length); j++)
				sids.add(unlabeled[j]);
			sentenceBatches[i] = sids.toNativeArray();
		}
	}
	
	public void project(double[] theta) throws InterruptedException
	{
		FirstOrderSequentialModelClone model = new FirstOrderSequentialModelClone(corpus.maxSequenceLength, 
				corpus.numStates, featureFunction);
		
		eta0 = config.initialLearningRate;
		eta = eta0;

		for(int i = 0; i < numSequences; i++) {
			PosSequence instance = (PosSequence) corpus.getInstance(i);
			
			if(instance.isLabeled) {
				for(int t = 0; t < instance.length; t++) {
					for(int k = 0; k < numStates; k++) {
						nodeMarginal[i][t][k] = (k == instance.tags[t]) ? 0 : Double.NEGATIVE_INFINITY;
						edgeMarginal[i][t][k][S0] = (t == 0 && k == instance.tags[0]) ? 0 : Double.NEGATIVE_INFINITY;
						for(int k2 = 0; k2 < numStates; k2++)
							edgeMarginal[i][t][k][k2] = (t > 0 && k == instance.tags[t] && 
								k2 == instance.tags[t-1]) ? 0 : Double.NEGATIVE_INFINITY;
					}
				}
			}
			else {
				model.computeScoresAndFB(instance, theta, backoff);
				oldLogNorm[i]= model.logNorm;

				for(int t = 0; t < instance.length; t++) 
					for(int k = 0; k < numStates; k++) {
						oldNodeScore[i][t][k] = model.nodeScore[t][k];
						nodeScore[i][t][k] = config.estepWarmstart ? oldNodeScore[i][t][k] : Math.log(uniformInit);
						if(t == 0) {
							oldEdgeScore[i][t][k][S0] = model.edgeScore[t][k][S0];
							edgeScore[i][t][k][S0] = config.estepWarmstart ? oldEdgeScore[i][t][k][S0] : Math.log(uniformInit);
						}
						else for(int k2 = 0; k2 < numStates; k2++) {
							oldEdgeScore[i][t][k][k2] = model.edgeScore[t][k][k2];
							edgeScore[i][t][k][k2] = config.estepWarmstart ? oldEdgeScore[i][t][k][k2] : Math.log(uniformInit);
						}						
					}
			}
		}
	
		double prevObjective = Double.NEGATIVE_INFINITY;
	
		for(int iter = 0; iter < numIterations; iter ++) {
		
			entropyObjective = 0;
			likelihoodObjective = 0;
			graphObjective = 0;
			double gradientNorm = 0;
			
			for(int i = 0; i < numThreads; i++) {
				uthreads[i] = new SentenceUpdateThread(sentenceBatches[i]);
				mthreads[i] = new SentenceMonitorThread(sentenceBatches[i], 
						new FirstOrderSequentialModelClone(corpus.maxSequenceLength, corpus.numStates, featureFunction),
						new EGDMonitor(numStates));
			}
		
			if(iter > 0) {	
				System.out.println("updating factorized edge scores ... ");
				for(int i = 0; i < numThreads;i ++) uthreads[i].start();
				for(int i = 0; i < numThreads;i ++) uthreads[i].join();
			}		
			
			for(int i = 0; i < numThreads;i ++) {
				gradientNorm += uthreads[i].gradientNorm;
				uthreads[i] = null;
			}
			
			System.out.println("updating edge marginals and node marginals ... ");
			
			for(int i = 0; i < numThreads;i ++) mthreads[i].start();
			for(int i = 0; i < numThreads;i ++) mthreads[i].join();
			
			double avgent = 0, avgmaxq = 0, avgstd = 0;
			double maxmaxq = Double.NEGATIVE_INFINITY, minmaxq = Double.POSITIVE_INFINITY;
			double acc = 0, norm = 0;
			int nid;
			
			for(int i = 0; i < numThreads; i++) {
				EGDMonitor monitor = mthreads[i].monitor;
				acc += monitor.numCorrect;
				norm += monitor.numTotal;
				avgent += monitor.avgent;
				avgmaxq += monitor.avgmaxq;
				avgstd += monitor.avgstd;
				maxmaxq = Math.max(maxmaxq, monitor.maxmaxq);
				minmaxq = Math.min(minmaxq, monitor.minmaxq);
			
				entropyObjective += mthreads[i].localEntropy;
				likelihoodObjective += mthreads[i].localLikelihood;
				mthreads[i] = null;
			}
			
			
			for(int sid = 0; sid < numSequences; sid++) {
				AbstractSequence instance = corpus.getInstance(sid);
				for(int t = 0; t < instance.length; t++) {
					if((nid = instance.nodes[t]) < 0) 
						continue;
					for(int k = 0; k < numStates; k++) {
						double nodeMar = Math.exp(nodeMarginal[sid][t][k]);
						for(int j = 0; j < graph.edges[nid].size(); j++) {
							int e = graph.edges[nid].get(j);
							double w = graph.weights[nid].get(j) / corpus.nodeFrequency[nid];
							graphObjective += w * nodeMar * (nodeMar - Math.exp(nodeMarginal[node2factor[e][0]][node2factor[e][1]][k]));
						}
					}
				}
			}
				
			double combinedObjective = klStrength * (likelihoodObjective - entropyObjective) + lpStrength * graphObjective;
			double objChange = Math.abs(combinedObjective - prevObjective);
			//if(iter % 10 == 9 ||  objChange < stoppingCriteria) {
				System.out.println(" ... " + (iter + 1) + " ... ");
				System.out.println("Negative Likelihood:\t" + likelihoodObjective + "\tEntropy:\t" + entropyObjective + 
						"\tGraph Violation:\t" + graphObjective + 
						"\tCombined objective:\t" + combinedObjective + "\tchange:\t" + objChange);
				System.out.println("Accuracy::\t" + acc / norm + "\tAvg entropy::\t" + avgent / norm + "\tAvg std::\t" + avgstd / norm);
				System.out.println("min max q::\t" + minmaxq + "\tmaxmaxq::\t" + maxmaxq + "\tavgmaxq::\t" + avgmaxq / norm);
				System.out.println(" gradient norm " + gradientNorm + " ... current eta: " + eta);
			
				if(objChange < stoppingCriteria) break;
			//}

			prevObjective = combinedObjective;
			
			eta = iter == 0 ? eta0 : eta0 / (1.0 + Math.sqrt(Math.sqrt(iter - 1)));

			sanityCheck();
		}
		
		if(config.hardEStep) {
			System.out.println("Making node marginals into hard counts");
			for(int sid : unlabeled) {
				AbstractSequence instance = corpus.getInstance(sid);
				int[] best = new int[instance.length];
				for(int t = 0; t < instance.length; t++) {
					best[t] = -1;
					for(int k = 0; k < numStates; k++) 
						if(best[t] < 0 || nodeMarginal[sid][t][k] > nodeMarginal[sid][t][best[t]]) 
							best[t] = k;
					for(int k = 0; k < numStates; k++) {
						nodeMarginal[sid][t][k] = (best[t] == k) ? 0 : Double.NEGATIVE_INFINITY;
						if(t == 0)
							edgeMarginal[sid][t][k][S0] = (best[t] == k) ? 0 : Double.NEGATIVE_INFINITY;
						else for(int k2 = 0; k2 < numStates; k2++)
							edgeMarginal[sid][t][k][k2] = (best[t] == k && best[t-1] == k2) ? 0 : Double.NEGATIVE_INFINITY;
					}
				}
			}
		}
		
		sanityCheck();

		//if(config.estepWarmstart) eta0 = eta;

		System.out.println("EDG Finished.");
	}
	
	private void sanityCheck()
	{
		double sum, sum0;
		
		for(int sid : unlabeled) {
			PosSequence instance = (PosSequence) corpus.getInstance(sid);
			for(int i = 0; i < instance.length; i++) {
				sum0 = 0;
				for(int s = 0; s < numStates; s++) {
					if(i > 0) {
						sum = 0;
						for(int sp = 0; sp < numStates; sp++) sum += Math.exp(edgeMarginal[sid][i][s][sp]);
						if(Math.abs(sum - Math.exp(nodeMarginal[sid][i][s])) > 1e-8) {
							System.out.println("[insanity] inconsistence with node and edge marginal: " + sid + ", " + i + ", " + s + ", " + sum);
							return;
						}
					}
					sum0 += Math.exp(nodeMarginal[sid][i][s]);
				}
				if(Math.abs(sum0 - 1.0) > 1e-8) {
					System.out.println("[insanity] node marginals not adding up to 1 " + sid + ", " + i + ", " + sum0);
					return;
				}
			}
		}
	}
	
	public double getNodeMarginal(int i, int t, int s)
	{
		return nodeMarginal[i][t][s];
	}
	
	public double getEdgeMarginal(int i, int t, int s, int sp)
	{
		return edgeMarginal[i][t][s][sp];
	}
	

	class SentenceUpdateThread extends Thread 
	{
		int[] sentenceIDs;
		double gradientNorm;
		
		public SentenceUpdateThread(int[] sentenceIDs) {
			this.sentenceIDs = sentenceIDs;
			this.gradientNorm = 0;
		}
		
		@Override
		public void run() {
			double grad;
			
			for(int sid : sentenceIDs) {
				PosSequence instance = (PosSequence) corpus.getInstance(sid);						

				for(int t = 0; t < instance.length; t++) 
					for(int j = 0; j < numStates; j++) {
						nodeScore[sid][t][j] -= eta * (grad = getNodeGradient(sid, t, j));
						gradientNorm += grad * grad;
						if(t == 0) {
							edgeScore[sid][t][j][S0] -= eta * (grad = getEdgeGradient(sid, t, j, S0));
							gradientNorm += grad * grad; 
						}
						else for(int k = 0; k < numStates; k++) {
							edgeScore[sid][t][j][k] -= eta * (grad = getEdgeGradient(sid, t, j, k));
							gradientNorm += grad * grad; 
						}			
					}			
			}
		}
		
		private double getEdgeGradient(int i, int t, int s, int sp)
		{
			return klStrength * (edgeScore[i][t][s][sp] - oldEdgeScore[i][t][s][sp]);
		}
		
		private double getNodeGradient(int i, int t, int s)
		{
			double gradA = nodeScore[i][t][s] - oldNodeScore[i][t][s];
			double gradB = 0;
			
			int nid = corpus.getInstance(i).nodes[t];
			if(nid >= 0) {
				double nodeMar = Math.exp(nodeMarginal[i][t][s]);			
				for(int j = 0; j < graph.edges[nid].size(); j++) {
					int e = graph.edges[nid].get(j); 
					double w = graph.weights[nid].get(j) / corpus.nodeFrequency[nid];
					gradB += w * ( nodeMar - Math.exp(nodeMarginal[node2factor[e][0]][node2factor[e][1]][s]) ) ;
				}
			}

			return klStrength * gradA + 2 * lpStrength * gradB;
		}
	}
	
	
	class SentenceMonitorThread extends Thread 
	{
		int[] sentenceIDs;
		FirstOrderSequentialModelClone model;
		EGDMonitor monitor;
		double localEntropy, localLikelihood;
		
		public SentenceMonitorThread(int[] sentenceIDs, FirstOrderSequentialModelClone model, EGDMonitor monitor) {
			this.sentenceIDs = sentenceIDs;
			this.model = model;
			this.monitor = monitor;
		}
		
		@Override
		public void run() {
			localEntropy = 0;
			localLikelihood = 0;
			
			for(int sid : sentenceIDs) {
				AbstractSequence instance = corpus.getInstance(sid);
				model.computeMarginals(edgeScore[sid], nodeScore[sid], edgeMarginal[sid], nodeMarginal[sid]);
				monitor.count(edgeScore[sid], nodeScore[sid], edgeMarginal[sid], nodeMarginal[sid], instance.tags);
				
				localEntropy += model.logNorm;
				localLikelihood += oldLogNorm[sid];
		
				for(int t = 0; t < instance.length; t++)
					for(int k = 0; k < numStates; k++) {
						if(t == 0) {
							double nodeMar = Math.exp(edgeMarginal[sid][t][k][S0]);
							localEntropy -=  nodeMar * (edgeScore[sid][t][k][S0] + nodeScore[sid][t][k]);
							localLikelihood -= nodeMar * (oldEdgeScore[sid][t][k][S0] + oldNodeScore[sid][t][k]);
						}
						else for(int k2 = 0; k2 < numStates; k2++) {
							double nodeMar = Math.exp(edgeMarginal[sid][t][k][k2]);
							localEntropy -=  nodeMar * (edgeScore[sid][t][k][k2] + nodeScore[sid][t][k]);
							localLikelihood -= nodeMar * (oldEdgeScore[sid][t][k][k2] + oldNodeScore[sid][t][k]);
						}
					}
			}
			
		}
	}
}
*/
