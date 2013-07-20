package constraints;

import analysis.FirstOrderEGDMonitor;
import analysis.SecondOrderEGDMonitor;
import models.AbstractFactorIterator;
import models.FirstOrderFactorGraph;
import models.MarginalsHelper;
import models.SecondOrderFactorGraph;
import models.UnconstrainedFactorIterator;
import config.Config;
import config.OCRConfig;
import data.AbstractCorpus;
import data.AbstractSequence;
import data.OCRCorpus;
import data.OCRSequence;
import data.SparseSimilarityGraph;
import features.FirstOrderPotentialFunction;
import features.OCRSOPotentialFunction;
import features.SecondOrderPotentialFunction;
import gnu.trove.TIntArrayList;

public class FirstOrderEG {

	double[][][] nodeMarginal, nodeScore, oldNodeScore;
	double[] oldLogNorm;
	double[] theta;
	
	int[][] decoded;
	int[][] node2factor;
	int[] unlabeled;
	int[][] sentenceBatches;
	int numUnlabeled;
	
	AbstractCorpus corpus;
	SparseSimilarityGraph graph;
	FirstOrderPotentialFunction ffunc;
	Config config;
	AbstractFactorIterator fiter;
	
	double lpStrength, klStrength;
	double backoff;
	double uniformInit;
	double eta0, eta, prodEta;
	int S0, SN;
	int numIterations, currIter;
	int numNodes, numSequences, numStates;
	int numThreads;
	
	SentenceUpdateThread[] uthreads;
	SentenceMonitorThread[] mthreads;
	
	public double entropyObjective, likelihoodObjective, graphObjective;

	double stoppingCriteria = 1e-5;
	double goldViolation;
	
	public FirstOrderEG(AbstractCorpus corpus, SparseSimilarityGraph graph, FirstOrderPotentialFunction ffunc, Config config)
	{
		this.corpus = corpus;
		this.graph = graph;
		this.ffunc = ffunc;
		this.config = config;
		this.fiter = new UnconstrainedFactorIterator(corpus);
		
		this.numNodes = corpus.numNodes;
		this.numSequences = corpus.numInstances;
		this.numStates = corpus.numTags;
		this.S0 = corpus.initialState;
		this.SN = corpus.finalState;
			
		this.numThreads = config.numThreads;
		this.uniformInit = config.estepInit;
		this.backoff = config.estepBackoff;
		this.lpStrength = config.graphRegularizationStrength;
		this.klStrength = 1.0 - lpStrength;
		
		this.numIterations = config.numEstepIters;
		
		System.out.println("Initializing EGD Contraint trainer ... ");
		System.out.println("Number of nodes:\t" + numNodes + "\tNumber of states\t" + numStates);
		
		System.out.println("****** gold violation of similarity graph ******");
		goldViolation = graph.computeGoldViolation(corpus);
		System.out.println("Gold violation::\t" + goldViolation);
		System.out.println("*****************************************\n");
		
		nodeScore = new double[numSequences][][];
		oldNodeScore = new double[numSequences][][];
		nodeMarginal = new double[numSequences][][];
		oldLogNorm = new double[numSequences];
		decoded = new int[numSequences][];
		node2factor = new int[numNodes][2];
		
		for(int i = 0; i < numSequences; i++) {
			AbstractSequence instance = corpus.getInstance(i);
			nodeScore[i] = new double[instance.length + 1][corpus.numStates];
			oldNodeScore[i] = new double[instance.length + 1][corpus.numStates];
			nodeMarginal[i] = new double[instance.length + 1][corpus.numStates];
			decoded[i] = new int[instance.length];
			for(int j = 0; j < instance.length; j++) {
				node2factor[instance.nodes[j]][0] = i;
				node2factor[instance.nodes[j]][1] = j;
			} 
		}
		
		unlabeled = corpus.unlabeled;
		numUnlabeled = unlabeled.length;
		
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
		this.eta0 = config.initialLearningRate;
		this.eta = eta0;
		this.prodEta = 1.0;	
		this.theta = theta;
		
		initializeCounts(theta);
		
		double prevObjective = Double.NEGATIVE_INFINITY;
	
		for(currIter = 0; currIter < numIterations; currIter ++) {
			
			entropyObjective = 0;
			likelihoodObjective = 0;
			graphObjective = computeGraphViolation();
			
			double gradientNorm = 0;
			
			for(int i = 0; i < numThreads; i++) {
				uthreads[i] = new SentenceUpdateThread(sentenceBatches[i]);
				mthreads[i] = new SentenceMonitorThread(sentenceBatches[i], 
						new FirstOrderFactorGraph(corpus, ffunc, fiter),
						new FirstOrderEGDMonitor(corpus.numStates, fiter));
			}
		
			if(currIter > 0) {	
				for(int i = 0; i < numThreads;i ++) uthreads[i].start();
				for(int i = 0; i < numThreads;i ++) uthreads[i].join();
				prodEta *= (1.0 - klStrength * eta);
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
				FirstOrderEGDMonitor monitor = mthreads[i].monitor;
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
			
			//if(currIter % 50 == 49 || objChange < stoppingCriteria) {
				System.out.println(" ... " + (currIter + 1) + " ... accumulated stepsize:\t" + prodEta);
				System.out.println("Negative Likelihood:\t" + likelihoodObjective + "\tEntropy:\t" + entropyObjective + 
						"\tGraph Violation:\t" + graphObjective + 
						"\tCombined objective:\t" + combinedObjective + "\tchange:\t" + objChange);
				System.out.println("Accuracy::\t" + acc / norm + "\tAvg entropy::\t" + avgent / norm + "\tAvg std::\t" + avgstd / norm);
				System.out.println("min max q::\t" + minmaxq + "\tmaxmaxq::\t" + maxmaxq + "\tavgmaxq::\t" + avgmaxq / norm);
				System.out.println("edge score range::\t" + mines + " - " + maxes + "\tnode score range::\t" + minns + " - " + maxns);
				System.out.println(" gradient norm " + gradientNorm + " ... current eta: " + eta);
			
				if(objChange < stoppingCriteria) break;
			//}

			prevObjective = combinedObjective;
			
			if(config.diminishingLearningRate) { 
				eta = currIter == 0 ? eta0 : eta0 / (1.0 + Math.sqrt(currIter - 1));
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
			/*
			 * EGDMonitor monitor = new SecondOrderEGDMonitor(corpus);
			for(int sid : unlabeled) {
				if(sid % 100 == 0) {
					System.out.println("\n ***" + sid);
					MarginalsHelper.printNodeMarginal(nodeMarginal[sid]);
				}
				MarginalsHelper.peakify(nodeMarginal[sid], edgeMarginal[sid], config.peakifyPower);
				monitor.count(edgeScore[sid], nodeScore[sid], edgeMarginal[sid], nodeMarginal[sid], corpus.getInstance(sid).tags);
				if(sid % 100 == 0) MarginalsHelper.printNodeMarginal(nodeMarginal[sid]);
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
			*/
		}
		
		//sanityCheck();
		//if(config.estepWarmstart) eta0 = eta;

		System.out.println("EDG Finished.");
	}
	

	private void initializeCounts(double[] theta)
	{
		FirstOrderFactorGraph model = new FirstOrderFactorGraph(corpus, ffunc, fiter);
		
		deepFill(nodeScore, Double.NEGATIVE_INFINITY);
		deepFill(oldNodeScore, Double.NEGATIVE_INFINITY);
		deepFill(nodeMarginal, Double.NEGATIVE_INFINITY);
		
		for(int sid = 0; sid < numSequences; sid++) {
			AbstractSequence instance = corpus.getInstance(sid);
			int length = instance.length;
			if(instance.isLabeled) {
				for(int i = 0; i < length; i++)
					for(int s : fiter.states(sid, i)) {
						nodeMarginal[sid][i][s] = (s == instance.tags[i]) ? 
								0 : Double.NEGATIVE_INFINITY;
					nodeMarginal[sid][length][SN] = 0;
				}
			}	
			else {
				model.computeScores(instance, theta, config.backoff);
				model.computeMarginals();
				oldLogNorm[sid]= model.logNorm;
				
				for(int i = 0; i <= length; i++)
					for(int s : fiter.states(sid, i)) {
						oldNodeScore[sid][i][s] = model.nodeScore[i][s];
						nodeScore[sid][i][s] = config.estepWarmstart ? 
								oldNodeScore[sid][i][s] : uniformInit;
					}
			}
		}
	}
	
	private double computeGraphViolation()
	{
		double gv = 0;
		for(int sid = 0; sid < numSequences; sid++) {
			AbstractSequence instance = corpus.getInstance(sid);
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
		/*
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
		*/
	}

	private void deepFill(double[][][] arr, double filler)
	{
		for(int i = 0; i < arr.length; i++)
			for(int j = 0; j < arr[i].length; j++)
				for(int k = 0; k < arr[i][j].length; k++) arr[i][j][k] = filler;
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
				AbstractSequence instance = corpus.getInstance(sid);						
				for(int i = 0; i <= instance.length; i++) 
					for(int s : fiter.states(sid, i)) {
						nodeScore[sid][i][s] -= eta * (grad = getNodeGradient(sid, i, s));
						gradientNorm += grad * grad;
					}			
			}
		}
		
		private double getNodeGradient(int sid, int i, int s)
		{
			double gradA = nodeScore[sid][i][s] - oldNodeScore[sid][i][s];
			double gradB = 0;
			
			if(s != SN && s != S0) { 
				int nid = corpus.getInstance(sid).nodes[i];
				double nodeMar = Math.exp(nodeMarginal[sid][i][s]);
				for(int j = 0; j < graph.edges[nid].size(); j++) {
					int e = graph.edges[nid].get(j);  
					double w = graph.weights[nid].get(j);
					gradB += w * (nodeMar - Math.exp(nodeMarginal[node2factor[e][0]][node2factor[e][1]][s]) ) ;
				}
			}

			return klStrength * gradA + 2 * lpStrength * gradB;
		}
	}
	
	public void projectScores(AbstractSequence instance, FirstOrderFactorGraph model)
	{
		int length = instance.length;
		int sid = instance.seqID;
		double Zt = prodEta;
		double Ztbar = 1.0 - Zt;
		
		for(int i = 0; i <= length; i++)
			for(int s : fiter.states(sid, i)) {
				model.nodeScore[i][s] = nodeScore[sid][i][s];
				if(!config.estepWarmstart) {
					for(int sp : fiter.states(sid, i-1)) { 
						model.edgeScore[i][s][sp] = 
							Zt * uniformInit + Ztbar * model.edgeScore[i][s][sp];
					}
				}
			}
	}
	
	class SentenceMonitorThread extends Thread 
	{
		int[] sentenceIDs;
		double[][][] oldScore;
		FirstOrderFactorGraph model;
		FirstOrderEGDMonitor monitor;
		double localEntropy, localLikelihood;
		
		public SentenceMonitorThread(int[] sentenceIDs, FirstOrderFactorGraph model, FirstOrderEGDMonitor monitor) {
			this.sentenceIDs = sentenceIDs;
			this.model = model;
			this.monitor = monitor;
			this.oldScore = new double[corpus.maxSequenceLength+1][corpus.numStates][corpus.numStates];
		}
		
		@Override
		public void run() {
			localEntropy = 0;
			localLikelihood = 0;
			
			for(int sid : sentenceIDs) {
				AbstractSequence instance = corpus.getInstance(sid);
				double fmar;
				
				model.computeScores(instance, theta, config.backoff);
				
				for(int i = 0; i <= instance.length; i++) 
					for(int s : fiter.states(sid, i)) 
						for(int sp : fiter.states(sid, i-1)) {
							oldScore[i][s][sp] = model.edgeScore[i][s][sp] + model.nodeScore[i][s];	
						}
						
				projectScores(instance, model);
				model.computeMarginals();
				
				monitor.count(instance, model, decoded[sid]);
				
				localEntropy += model.logNorm;
				localLikelihood += oldLogNorm[sid];
		
				for(int i = 0; i <= instance.length; i++) 
					for(int s : fiter.states(sid, i)) { 
						nodeMarginal[sid][i][s] = model.nodeMarginal[i][s];
						for(int sp : fiter.states(sid, i-1)) {
							fmar = Math.exp(model.edgeMarginal[i][s][sp]);
							localEntropy -=  fmar * (model.edgeScore[i][s][sp] + model.nodeScore[i][s]);
							localLikelihood -= fmar * oldScore[i][s][sp];
						}
					}
				
			}
		}
	}
	
}
