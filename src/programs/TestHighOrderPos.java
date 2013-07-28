package programs;

import features.PosSOPotentialFunction;
import java.io.IOException;
import java.util.Random;
import models.AbstractFactorIterator;
import models.PrunedTagIterator;
import config.Config;
import config.PosConfig;
import trainers.SecondOrderEMTrainer;
import util.MemoryTracker;
import data.NGramMapper;
import data.PosCorpus;
import data.SparseSimilarityGraph;

public class TestHighOrderPos {
	private static void resampleTrains(Config config, PosCorpus corpus) {
		int fsize = config.numLabels;
		corpus.sampleFromFolderUnsorted(fsize * config.numCVFolds,
				config.seedFolder, new Random(12345));
		
		int[] newTests = new int[corpus.numInstances - fsize];
		int[] newTrains = new int[fsize];
		
		int dsize = 0;
		for (int i = 0; i < corpus.tests.length; i++) {
			newTests[dsize++] = corpus.tests[i];
		}
		
		for (int k = 0; k < config.numCVFolds; k++) {
			for (int i = 0; i < fsize; i++) {
				int tid = k * fsize + i;
				if (k == config.sampleFoldID) {
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
		PosConfig config = new PosConfig(args);
		config.print(System.out);			

		MemoryTracker mem  = new MemoryTracker();
		mem.start(); 
		
		String[] dataFiles = new String[] {
				config.dataPath + ".train.ulab", 
				config.dataPath + ".test.ulab"};
		
		NGramMapper ngmap = new NGramMapper(config);
		PosCorpus corpus = new PosCorpus(dataFiles, ngmap, config);
		resampleTrains(config, corpus);

		SparseSimilarityGraph graph = null;
		try {
			graph = new SparseSimilarityGraph(config.graphPath, corpus.numNodes);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		PosSOPotentialFunction potentialFunction = new PosSOPotentialFunction(
				corpus, config);
		AbstractFactorIterator fiter = new PrunedTagIterator(corpus);
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
