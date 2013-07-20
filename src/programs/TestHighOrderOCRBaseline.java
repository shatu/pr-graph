package programs;

import features.OCRSOPotentialFunction;
import java.io.IOException;
import java.util.Random;

import models.AbstractFactorIterator;
import models.UnconstrainedFactorIterator;
import config.OCRConfig;
import trainers.GraphBootstrappedTrainer;
import util.MemoryTracker;

import data.OCRCorpus;
import data.SparseSimilarityGraph;

public class TestHighOrderOCRBaseline {

	static int maxNumNodes = Integer.MAX_VALUE;
    static int maxNumSampleFolds = 10;

    private static void resampleTrains(OCRConfig config, OCRCorpus corpus)
    {
        int fsize = config.numLabels;
        corpus.sampleFromFolderUnsorted(fsize * maxNumSampleFolds, config.seedFolder, config.holdoutFolder, new Random(12345));

        int[] newTests = new int[corpus.tests.length];
        int[] newDevs = new int[corpus.numInstances - fsize - corpus.tests.length];
        int[] newTrains = new int[fsize];

        for(int i = 0; i < newTests.length; i++)
            newTests[i] = corpus.tests[i];

        int dsize = 0;
        for(int i = 0; i < corpus.devs.length; i++)
            newDevs[dsize++] = corpus.devs[i];

        for(int k = 0; k < maxNumSampleFolds; k++) {
            for(int i = 0; i < fsize; i++) {
                int tid = k * fsize + i;
                if(k == config.sampleFoldID)
                    newTrains[i] = corpus.trains[tid];
                else
                    newDevs[dsize++] = corpus.trains[tid];
            }
        }
        corpus.resetLabels(newTrains, newDevs, newTests);
        corpus.printCrossValidationInfo();
    }
	
	public static void main(String[] args) throws NumberFormatException, IOException 
	{
		OCRConfig config = new OCRConfig(args);
		config.print(System.out);			

		MemoryTracker mem  = new MemoryTracker();
		mem.start(); 
		
		OCRCorpus corpus = new OCRCorpus(config.dataPath, maxNumNodes);
		resampleTrains(config, corpus);

		SparseSimilarityGraph graph = null;
		try {
			graph = new SparseSimilarityGraph(config.graphPath, corpus.numNodes, false);
		} catch (Exception e) {
			e.printStackTrace();
		}
		
		OCRSOPotentialFunction potentialFunction = new OCRSOPotentialFunction(corpus, config);
		AbstractFactorIterator fiter = new UnconstrainedFactorIterator(corpus);
		GraphBootstrappedTrainer trainer = new GraphBootstrappedTrainer(corpus, potentialFunction, graph, fiter, config);
	
		trainer.trainModel();
		
		System.out.println("Training accuracy::");
		trainer.testModel(corpus.trains);
		
		System.out.println("Dev accuracy::");
		trainer.testModel(corpus.devs);
		
		System.out.println("Testing accuracy::");
		trainer.testModel(corpus.tests);
		
		System.out.println("Unlabeled accuracy::");
		trainer.testModel(corpus.unlabeled);
		
		System.out.print("All unlabeled accuracy::\t");
		trainer.testAndAnalyze(corpus.unlabeled, "pr");
	
		mem.finish();
		System.out.println("Memory usage:: " + mem.print());
	}

}
