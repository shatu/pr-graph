package programs;

import features.OCRSOPotentialFunction;
import java.io.IOException;
import java.util.Random;
import models.AbstractFactorIterator;
import models.UnconstrainedFactorIterator;
import config.Config;
import config.OCRConfig;
import trainers.SecondOrderEMTrainer;
import util.MemoryTracker;
import data.OCRCorpus;
import data.SparseSimilarityGraph;

public class TestHighOrderOCR {

	static int maxNumNodes = Integer.MAX_VALUE;
	static int maxNumSampleFolds = 10;
	
	private static void resampleTrains(Config config, OCRCorpus corpus) {
		int fsize = config.numLabels;
		corpus.sampleFromFolderUnsorted(fsize * maxNumSampleFolds,
				config.seedFolder, new Random(12345));
		
		int[] newTests = new int[corpus.numInstances - fsize];
		int[] newTrains = new int[fsize];
		
		int dsize = 0;
		for(int i = 0; i < corpus.tests.length; i++) {
			newTests[dsize++] = corpus.tests[i];
		}
		
		for(int k = 0; k < maxNumSampleFolds; k++) {
			for(int i = 0; i < fsize; i++) {
				int tid = k * fsize + i;
				if(k == config.sampleFoldID) {
					newTrains[i] = corpus.trains[tid];
				}
				else { 
					newTests[dsize++] = corpus.trains[tid];
				}
			}
		}
		corpus.resetLabels(newTrains, newTests);
		corpus.printCrossValidationInfo();
	}
	
	public static void main(String[] args)
			throws NumberFormatException, IOException {
		
		OCRConfig config = new OCRConfig(args);
		config.print(System.out);			

		MemoryTracker mem  = new MemoryTracker();
		mem.start(); 
		
		OCRCorpus corpus = new OCRCorpus(config.dataPath, maxNumNodes);
		resampleTrains(config, corpus);
		
		SparseSimilarityGraph graph = null;
		try {
			graph = new SparseSimilarityGraph(config.graphPath, corpus.numNodes,
					false);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		OCRSOPotentialFunction potentialFunction = new OCRSOPotentialFunction(
				corpus, config);
		AbstractFactorIterator fiter = new UnconstrainedFactorIterator(corpus);
		SecondOrderEMTrainer trainer = new SecondOrderEMTrainer(corpus,
				potentialFunction, graph, fiter, config);
	
		trainer.trainModel();
		
		System.out.print("Training accuracy::\t");
		trainer.testModel(corpus.trains);
		
		System.out.print("Testing accuracy::\t");
		trainer.testModel(corpus.tests);
		
		mem.finish();
		System.out.println("Memory usage:: " + mem.print());
	}

}
