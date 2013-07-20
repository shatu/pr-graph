package programs;

import features.PosSOPotentialFunction;
import java.io.IOException;
import java.util.Random;
import models.AbstractFactorIterator;
import models.PrunedTagIterator;
import config.PosConfig;
import config.Config;
import trainers.GraphBootstrappedTrainer;
import util.MemoryTracker;
import data.NGramMapper;
import data.PosCorpus;
import data.SparseSimilarityGraph;

public class TestHighOrderPosBaseline {

	static int maxNumNodes = Integer.MAX_VALUE;
	static int maxNumSampleFolds = 10;
	
	private static void resampleTrains(Config config, PosCorpus corpus)
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
		PosConfig config = new PosConfig(args);
		config.print(System.out);			

		MemoryTracker mem  = new MemoryTracker();
		mem.start(); 
		
		String[] dataFiles = new String[] {
				config.dataPath + ".train.ulab", 
				config.dataPath + ".test.ulab"};
		
		NGramMapper ngmap = new NGramMapper(config.ngramPath);
		PosCorpus corpus = new PosCorpus(dataFiles, ngmap, config);
		resampleTrains(config, corpus);


		SparseSimilarityGraph graph = null;
		try {
			graph = new SparseSimilarityGraph(config.graphPath, corpus.numNodes, false);
		} catch (Exception e) {
			e.printStackTrace();
		}
	
		PosSOPotentialFunction potentialFunction = new PosSOPotentialFunction(corpus, config);
		AbstractFactorIterator fiter = new PrunedTagIterator(corpus);
		GraphBootstrappedTrainer trainer = new GraphBootstrappedTrainer(corpus, potentialFunction, graph, fiter, config);
		
		trainer.trainModel();
			
		System.out.print("Training accuracy::\t");
		trainer.testModel(corpus.trains);
			
		System.out.print("Dev accuracy::\t");
		trainer.testModel(corpus.devs);
			
		System.out.print("Testing accuracy::\t");
		trainer.testModel(corpus.tests);
			
		System.out.print("All unlabeled accuracy::\t");
		trainer.testModel(corpus.unlabeled);
	
	
		mem.finish();
		System.out.println("Memory usage:: " + mem.print());
	}

}
