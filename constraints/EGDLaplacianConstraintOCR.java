package constraints;

import gnu.trove.TIntArrayList;

import java.util.Random;

import config.OCRConfig;
import analysis.FirstOrderEGDMonitor;
import models.MarginalsHelper;

import data.OCRCorpus;
import data.OCRSequence;
import data.SparseSimilarityGraph;

/*********
 * 
 * Use seperate node potential and edge potential 
 * 	
 * @author luhe
 *
 */

/*
public class EGDLaplacianConstraintOCR implements AbstractEGDConstraint
{	
	
	double[][][][] edgeScore, oldEdgeScore, edgeMarginal;
	double[][][] nodeScore, oldNodeScore, nodeMarginal; 
	double[] oldLogNorm;
	
	int[][] node2factor;
	int[] unlabeled;
	int[][] sentenceBatches;
	int numUnlabeled;
	
	OCRCorpus corpus;
	SparseSimilarityGraph graph;
	OCRFeatureFunction featureFunction;
	OCRConfig config;
	
	double lpStrength, klStrength;
	double backoff;
	double uniformInit;
	double eta0, eta;
	int S0, SN;
	int numIterations;
	
	int numNodes, numSequences, numStates;
	
	int numThreads;
	Random sentencePicker;

	SentenceUpdateThread[] uthreads;
	SentenceMonitorThread[] mthreads;
	
	public double entropyObjective, likelihoodObjective, graphObjective;

	double stoppingCriteria = 1e-5;
	double goldViolation;
	
	public EGDLaplacianConstraintOCR(OCRCorpus corpus, SparseSimilarityGraph graph, OCRFeatureFunction featureFunction, OCRConfig config)
	{
		this.corpus = corpus;
		this.graph = graph;
		this.featureFunction = featureFunction;
		this.config = config;
		
		this.numNodes = corpus.numNodes;
		this.numSequences = corpus.numInstances;
		this.numStates = corpus.numStates - 2;
		this.S0 = corpus.initialState;
		this.SN = corpus.finalState;
		
		this.eta0 = config.initialLearningRate;
		this.eta = eta0;
		
		this.numThreads = config.numThreads;
		this.uniformInit = config.estepInit;
		this.backoff = 0; //config.backoff;
		this.lpStrength = config.graphRegularizationStrength;
		this.klStrength = 1.0 - lpStrength;
		
		this.numIterations = config.numEstepIters;
		
		System.out.println("Initializing EGD Contraint trainer ... ");
		System.out.println("Number of nodes:\t" + numNodes + "\tNumber of states\t" + numStates);
		
		System.out.println("****** gold violation of similarity graph ******");
		goldViolation = graph.computeGoldViolation(corpus);
		System.out.println("Gold violation::\t" + goldViolation);
		System.out.println("*****************************************\n");
		
		edgeScore = new double[numSequences][][][];
		oldEdgeScore = new double[numSequences][][][];
		edgeMarginal = new double[numSequences][][][];
		
		nodeScore = new double[numSequences][][];
		oldNodeScore = new double[numSequences][][];
		nodeMarginal = new double[numSequences][][];
		
		oldLogNorm = new double[numSequences];
		node2factor = new int[numNodes][2];
		
		for(int i = 0; i < numSequences; i++) {
			OCRSequence instance = corpus.getInstance(i);
			edgeScore[i] = new double[instance.length + 1][numStates + 2][numStates + 1];
			oldEdgeScore[i] = new double[instance.length + 1][numStates + 2][numStates + 1];
			edgeMarginal[i] = new double[instance.length + 1][numStates + 2][numStates + 1];
			
			nodeScore[i] = new double[instance.length + 1][numStates+2];
			oldNodeScore[i] = new double[instance.length + 1][numStates+2];
			nodeMarginal[i] = new double[instance.length + 1][numStates+2];
			
			for(int j = 0; j < instance.length; j++) {
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
		
		eta0 = config.initialLearningRate;
		eta = eta0;
		
		initializeCounts(theta);
		
		double prevObjective = Double.NEGATIVE_INFINITY;
	
		for(int iter = 0; iter < numIterations; iter ++) {
			
			entropyObjective = 0;
			likelihoodObjective = 0;
			graphObjective = computeGraphViolation();
			
			double gradientNorm = 0;
			
			for(int i = 0; i < numThreads; i++) {
				uthreads[i] = new SentenceUpdateThread(sentenceBatches[i]);
				mthreads[i] = new SentenceMonitorThread(sentenceBatches[i], 
						new FirstOrderSequentialModelClone(corpus.maxSequenceLength, corpus.numStates, featureFunction),
						new EGDMonitor(numStates));
			}
		
			if(iter > 0) {	
				for(int i = 0; i < numThreads;i ++) uthreads[i].start();
				for(int i = 0; i < numThreads;i ++) uthreads[i].join();
			}		
			
			for(int i = 0; i < numThreads;i ++) {
				gradientNorm += uthreads[i].gradientNorm;
				uthreads[i] = null;
			}
			
			for(int i = 0; i < numThreads;i ++) mthreads[i].start();
			for(int i = 0; i < numThreads;i ++) mthreads[i].join();
			
			double avgent = 0, avgmaxq = 0, avgstd = 0;
			double maxmaxq = Double.NEGATIVE_INFINITY, minmaxq = Double.POSITIVE_INFINITY;
			double mines = Double.POSITIVE_INFINITY, maxes = Double.NEGATIVE_INFINITY;
			double minns = Double.POSITIVE_INFINITY, maxns = Double.NEGATIVE_INFINITY;
			double acc = 0, norm = 0;
			
			for(int i = 0; i < numThreads; i++) {
				EGDMonitor monitor = mthreads[i].monitor;
				acc += monitor.numCorrect;
				norm += monitor.numTotal;
				avgent += monitor.avgent;
				avgmaxq += monitor.avgmaxq;
				avgstd += monitor.avgstd;
				maxmaxq = Math.max(maxmaxq, monitor.maxmaxq);
				minmaxq = Math.min(minmaxq, monitor.minmaxq);
			
				minns = Math.min(minns, monitor.nsrange[0]);
				maxns = Math.max(maxns, monitor.nsrange[1]);
				mines = Math.min(mines, monitor.esrange[0]);
				maxes = Math.max(maxes, monitor.esrange[1]);
				
				entropyObjective += mthreads[i].localEntropy;
				likelihoodObjective += mthreads[i].localLikelihood;
				mthreads[i] = null;
			}
				
			double combinedObjective = klStrength * (likelihoodObjective - entropyObjective) + lpStrength * graphObjective;
			double objChange = Math.abs(combinedObjective - prevObjective);
			
			if(iter % 50 == 49 || objChange < stoppingCriteria) {
				System.out.println(" ... " + (iter + 1) + " ... ");
				System.out.println("Negative Likelihood:\t" + likelihoodObjective + "\tEntropy:\t" + entropyObjective + 
						"\tGraph Violation:\t" + graphObjective + 
						"\tCombined objective:\t" + combinedObjective + "\tchange:\t" + objChange);
				System.out.println("Accuracy::\t" + acc / norm + "\tAvg entropy::\t" + avgent / norm + "\tAvg std::\t" + avgstd / norm);
				System.out.println("min max q::\t" + minmaxq + "\tmaxmaxq::\t" + maxmaxq + "\tavgmaxq::\t" + avgmaxq / norm);
				System.out.println("edge score range::\t" + mines + " - " + maxes + "\tnode score range::\t" + minns + " - " + maxns);
				System.out.println(" gradient norm " + gradientNorm + " ... current eta: " + eta);
			
				if(objChange < stoppingCriteria) break;
			}

			prevObjective = combinedObjective;
			
			if(config.diminishingLearningRate) { 
				eta = iter == 0 ? eta0 : eta0 / (1.0 + Math.sqrt(iter - 1));
			}
		}
		
		if(config.hardEStep) {
			System.out.println("Making node marginals into hard counts");
			makeHardCounts();
			System.out.println("Graph violation on hard counts:\t" + computeGraphViolation() + 
					"\tgold:\t" + goldViolation);
		}
		else if(config.peakifyPower > 1) {
			System.out.println("Trying to make the probability peaked, raising to the power " + config.peakifyPower);
			EGDMonitor monitor = new EGDMonitor(numStates);
			for(int sid : unlabeled) {
				//if(sid % 100 == 0) {
				//	System.out.println("\n ***" + sid);
				//	MarginalsHelper.printNodeMarginal(nodeMarginal[sid]);
				//}
				MarginalsHelper.peakify(nodeMarginal[sid], edgeMarginal[sid], config.peakifyPower);
				monitor.count(edgeScore[sid], nodeScore[sid], edgeMarginal[sid], nodeMarginal[sid], corpus.getInstance(sid).tags);
				
				//if(sid % 100 == 0) MarginalsHelper.printNodeMarginal(nodeMarginal[sid]);
			}
			System.out.println("After peakify ...");
			System.out.println("Accuracy::\t" + 1.0 * monitor.numCorrect / monitor.numTotal + "\tAvg entropy::\t" + 
					monitor.avgent / monitor.numTotal + "\tAvg std::\t" + 
					monitor.avgstd / monitor.numTotal);
			System.out.println("min max q::\t" + monitor.minmaxq + "\tmaxmaxq::\t" + 
					monitor.maxmaxq + "\tavgmaxq::\t" + 
					monitor.avgmaxq / monitor.numTotal);
			
			System.out.println("Graph violation after peakify:\t" + computeGraphViolation() + 
					"\tgold:\t" + goldViolation);
		}
		
		//sanityCheck();
		//if(config.estepWarmstart) eta0 = eta;

		System.out.println("EDG Finished.");
	}
	

	private void initializeCounts(double[] theta)
	{
		FirstOrderSequentialModelClone model = new FirstOrderSequentialModelClone(corpus.maxSequenceLength, 
				corpus.numStates, featureFunction);
		
		double init = Math.log(uniformInit);
		double smo = Math.log(config.estepBackoff);
		
		deepFill(nodeScore, Double.NEGATIVE_INFINITY);
		deepFill(edgeScore, Double.NEGATIVE_INFINITY);
		deepFill(oldNodeScore, Double.NEGATIVE_INFINITY);
		deepFill(oldEdgeScore, Double.NEGATIVE_INFINITY);
		deepFill(nodeMarginal, Double.NEGATIVE_INFINITY);
		deepFill(edgeMarginal, Double.NEGATIVE_INFINITY);

		for(int i = 0; i < numSequences; i++) {
			OCRSequence instance = corpus.getInstance(i);
			int len = instance.length;
			
			if(instance.isLabeled) {
				for(int t = 0; t < len; t++) {
					for(int k = 0; k < numStates; k++) {
						nodeMarginal[i][t][k] = (k == instance.tags[t]) ? 
								0 : Double.NEGATIVE_INFINITY;
						edgeMarginal[i][t][k][S0] = (t == 0 && k == instance.tags[0]) ? 
								0 : Double.NEGATIVE_INFINITY;
						edgeMarginal[i][t][SN][k] = (t == len && k == instance.tags[len - 1]) ? 
								0 : Double.NEGATIVE_INFINITY;
						
						for(int k2 = 0; k2 < numStates; k2++)
							edgeMarginal[i][t][k][k2] = (t > 0 && k == instance.tags[t] && k2 == instance.tags[t-1]) ? 
									0 : Double.NEGATIVE_INFINITY;
					}
				}
				
			}
			else {
				model.computeScoresAndFB(instance, theta, backoff);
				if(config.hardMStep) 
					model.decodeAndEvaluate(instance.tags);
				
				oldLogNorm[i]= model.logNorm;
				
				oldNodeScore[i][len][SN] = model.nodeScore[len][SN];
				nodeScore[i][len][SN] = config.estepWarmstart ? 
						oldNodeScore[i][len][SN] : init;
				
				for(int k = 0; k < numStates; k++) {
					oldEdgeScore[i][0][k][S0] = (!config.hardMStep || model.decode[0] == k) ?
							model.edgeScore[0][k][S0] : smo;
							
					edgeScore[i][0][k][S0] = config.estepWarmstart ? 
							oldEdgeScore[i][0][k][S0] : init;
							
					oldEdgeScore[i][len][SN][k] = (!config.hardMStep || model.decode[len-1] == k) ? 
							model.edgeScore[len][SN][k] : smo;
					
					edgeScore[i][len][SN][k] = config.estepWarmstart ? 
								oldEdgeScore[i][len][SN][k] : init;
				}
				
				for(int t = 0; t < len; t++) 
					for(int k = 0; k < numStates; k++) {
						oldNodeScore[i][t][k] = (!config.hardMStep || model.decode[t] == k) ? 
								model.nodeScore[t][k] : smo;
								
						nodeScore[i][t][k] = config.estepWarmstart ? 
								oldNodeScore[i][t][k] : init;
				
						if(t > 0)
							for(int k2 = 0; k2 < numStates; k2++) {
								oldEdgeScore[i][t][k][k2] = (!config.hardMStep || (model.decode[t] == k && model.decode[t-1] == k2)) ?
										model.edgeScore[t][k][k2] : smo;
										
								edgeScore[i][t][k][k2] = config.estepWarmstart ? 
										oldEdgeScore[i][t][k][k2] : init;
							}						
					}
			}
		}
	}
	
	private double computeGraphViolation()
	{
		double gv = 0;
		for(int sid = 0; sid < numSequences; sid++) {
			OCRSequence instance = corpus.getInstance(sid);
			for(int t = 0; t < instance.length; t++) {
				int nid = instance.nodes[t];
				for(int k = 0; k < numStates; k++) {
					double nodeMar = Math.exp(nodeMarginal[sid][t][k]);
					for(int j = 0; j < graph.edges[nid].size(); j++) {
						int e = graph.edges[nid].get(j);
						double w = graph.weights[nid].get(j);
						gv += w * nodeMar * (nodeMar - Math.exp(nodeMarginal[node2factor[e][0]][node2factor[e][1]][k]));
					}
				}
			}
		}
		return gv;
	}
	
	private void makeHardCounts()
	{
		System.out.println("Making node marginals into hard counts");
		for(int sid : unlabeled) {
			OCRSequence instance = corpus.getInstance(sid);
			int length = instance.length;
			int[] best = new int[length];
			
			for(int t = 0; t < length; t++) {
				best[t] = -1;
				for(int k = 0; k < numStates; k++) 
					if(best[t] < 0 || nodeMarginal[sid][t][k] > nodeMarginal[sid][t][best[t]]) 
						best[t] = k;
				
				for(int k = 0; k < numStates; k++) {
					nodeMarginal[sid][t][k] = (best[t] == k) ? 0 : Double.NEGATIVE_INFINITY;			
					if(t > 0) for(int k2 = 0; k2 < numStates; k2++)
						edgeMarginal[sid][t][k][k2] = (best[t] == k && best[t-1] == k2) ? 0 : Double.NEGATIVE_INFINITY;
				}
			}
			
			for(int k = 0; k < numStates; k++) {
				edgeMarginal[sid][0][k][S0] = (best[0] == k) ? 0 : Double.NEGATIVE_INFINITY;
				edgeMarginal[sid][length][SN][k] = (best[length-1] == k) ? 0 : Double.NEGATIVE_INFINITY;
			}
		}
	}

	private void deepFill(double[][][] arr, double filler)
	{
		for(int i = 0; i < arr.length; i++)
			for(int j = 0; j < arr[i].length; j++)
				for(int k = 0; k < arr[i][j].length; k++) arr[i][j][k] = filler;
	}
	
	private void deepFill(double[][][][] arr, double filler) 
	{
		for(int i = 0; i < arr.length; i++) deepFill(arr[i], filler);
	}
	
	@Override
	public double getNodeMarginal(int i, int t, int s)
	{
		return nodeMarginal[i][t][s];
	}
	
	@Override
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
				OCRSequence instance = corpus.getInstance(sid);						
				int len = instance.length;
				
				for(int j = 0; j < numStates; j++) {
					edgeScore[sid][0][j][S0] -= eta * (grad = getEdgeGradient(sid, 0, j, S0));
					gradientNorm += grad * grad;
					
					edgeScore[sid][len][SN][j] -= eta * (grad = getEdgeGradient(sid, len, SN, j));
					gradientNorm += grad * grad;
				}
				
				nodeScore[sid][len][SN] -= eta * (grad = getNodeGradient(sid, len, SN));
				gradientNorm += grad * grad;
				
				for(int t = 0; t < instance.length; t++) 
					for(int j = 0; j < numStates; j++) {
						nodeScore[sid][t][j] -= eta * (grad = getNodeGradient(sid, t, j));
						gradientNorm += grad * grad;
						if(t > 0) 
							for(int k = 0; k < numStates; k++) {
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
			
			if(s != SN) { // if not final state
				int nid = corpus.getInstance(i).nodes[t];
				double nodeMar = Math.exp(nodeMarginal[i][t][s]);
				
				for(int j = 0; j < graph.edges[nid].size(); j++) {
					int e = graph.edges[nid].get(j); // is another node ... 
					double w = graph.weights[nid].get(j);
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
				OCRSequence instance = corpus.getInstance(sid);
				int len = instance.length;
				double nodeMar;
				
				model.computeMarginals(edgeScore[sid], nodeScore[sid], edgeMarginal[sid], nodeMarginal[sid]);
				monitor.count(edgeScore[sid], nodeScore[sid], edgeMarginal[sid], nodeMarginal[sid], corpus.getInstance(sid).tags);
				
				localEntropy += model.logNorm;
				localLikelihood += oldLogNorm[sid];
		
				for(int k = 0; k < numStates; k++) {
					nodeMar = Math.exp(edgeMarginal[sid][0][k][S0]);
					localEntropy -=  nodeMar * (edgeScore[sid][0][k][S0] + nodeScore[sid][0][k]);
					localLikelihood -= nodeMar * (oldEdgeScore[sid][0][k][S0] + oldNodeScore[sid][0][k]);
					
					nodeMar = Math.exp(edgeMarginal[sid][len][SN][k]);
					localEntropy -=  nodeMar * (edgeScore[sid][len][SN][k] + nodeScore[sid][len][SN]);
					localLikelihood -= nodeMar * (oldEdgeScore[sid][len][SN][k] + oldNodeScore[sid][len][SN]);
				}
				
				for(int t = 1; t < instance.length; t++)
					for(int k = 0; k < numStates; k++) {
						for(int k2 = 0; k2 < numStates; k2++) {
							nodeMar = Math.exp(edgeMarginal[sid][t][k][k2]);
							localEntropy -=  nodeMar * (edgeScore[sid][t][k][k2] + nodeScore[sid][t][k]);
							localLikelihood -= nodeMar * (oldEdgeScore[sid][t][k][k2] + oldNodeScore[sid][t][k]);
						}
					}
			}
			
		}
	}
	
}
*/
