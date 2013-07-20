package programs;

import features.OCRFOPotentialFunction;
import java.io.IOException;
import java.util.Random;

import models.AbstractFactorIterator;
import models.UnconstrainedFactorIterator;
import config.OCRConfig;
import trainers.FirstOrderEMTrainer;
import util.MemoryTracker;

import data.OCRCorpus;
import data.SparseSimilarityGraph;

public class TestFirstOrderOCR {

	static int maxNumNodes = Integer.MAX_VALUE;
	
	public static void main(String[] args) throws NumberFormatException, IOException 
	{
		OCRConfig config = new OCRConfig(args);
		config.print(System.out);			

		MemoryTracker mem  = new MemoryTracker();
		mem.start(); 
		
		OCRCorpus corpus = new OCRCorpus(config.dataPath, maxNumNodes);
		corpus.sampleFromFolder(config.numLabels, config.seedFolder, config.holdoutFolder, new Random(12345));	
		SparseSimilarityGraph graph = null;
		try {
			graph = new SparseSimilarityGraph(config.graphPath, corpus.numNodes, false);
		} catch (Exception e) {
			e.printStackTrace();
		}

		OCRFOPotentialFunction potentialFunction = new OCRFOPotentialFunction(corpus, config);
		AbstractFactorIterator fiter = new UnconstrainedFactorIterator(corpus);
		FirstOrderEMTrainer trainer = new FirstOrderEMTrainer(corpus, potentialFunction, graph, fiter, config);
	
		trainer.trainModel();
		
		System.out.print("Training accuracy::\t");
		trainer.testModel(corpus.trains);
		
		System.out.print("Dev accuracy::\t");
		trainer.testModel(corpus.devs);
		
		System.out.print("Testing accuracy::\t");
		trainer.testModel(corpus.tests);
		
		System.out.print("All unlabeled accuracy::\t");
		trainer.testAndAnalyze(corpus.unlabeled, "gp-crf");
	
		mem.finish();
		System.out.println("Memory usage:: " + mem.print());
	}

}
