package programs;

import graph.PosGraphBuilder;
import java.io.IOException;
import util.MemoryTracker;
import config.PosGraphConfig;
import data.PosCorpus;

public class TestPosGraphBuilder {

	public static void main(String[] args) throws NumberFormatException,
			IOException	{
		PosGraphConfig config = new PosGraphConfig(args);
		config.print(System.out);			

		MemoryTracker mem  = new MemoryTracker();
		mem.start(); 
		
		String[] dataFiles = new String[] {
				config.dataPath + ".train.ulab", 
				config.dataPath + ".test.ulab"};
		
		PosCorpus corpus = new PosCorpus(dataFiles, null, config);
		
		PosGraphBuilder builder = new PosGraphBuilder(corpus, config);
		builder.buildGraph();
		
		mem.finish();
		System.out.println("Memory usage:: " + mem.print());
	}
}
