package data;

import java.io.PrintStream;
import java.util.Arrays;
import java.util.Random;
import gnu.trove.TIntArrayList;

public class AbstractCorpus {
	public int[] trains, devs, tests, unlabeled;
	public int maxSequenceID, maxSequenceLength;
	public int numStates, numTags, initialState, finalState, initialStateSO;
	public int numInstances, numNodes, numWords;
	
	public int[] nodeFrequency;
	
	public AbstractSequence getInstance(int id) {
		return null;
	}
	
	public void printInstance(int sid, PrintStream ostr) {
		//FIXME: to implement
	}
	
	public void sampleFromFolder(int numLabels, int seedFolder,
			int holdoutFolder, Random sampler) {
		sampleFromFolderUnsorted(numLabels, seedFolder, holdoutFolder, sampler);
		Arrays.sort(trains);
		Arrays.sort(devs);
		Arrays.sort(tests);
		Arrays.sort(unlabeled);
	}
	
	public void sampleFromFolderUnsorted(int numLabels, int seedFolder,
			int holdoutFolder, Random sampler) {
		TIntArrayList poolIds = new TIntArrayList();
		TIntArrayList trainIds = new TIntArrayList();
		TIntArrayList testIds = new TIntArrayList();
		TIntArrayList devIds = new TIntArrayList();
		TIntArrayList unlabeledIds = new TIntArrayList();
		
		for(int i = 0; i < numInstances; i++) {
			int fid = getInstance(i).foldID; 
			if(fid == seedFolder)
				poolIds.add(i);
			else if(fid == holdoutFolder) {
				testIds.add(i);
				getInstance(i).isHeldout = true;
			}
		}
		
		int[] pool = poolIds.toNativeArray();
		boolean[] sampled = new boolean[pool.length];
		for(int i = 0; i < pool.length; i++) 
			sampled[i] = false;
		
		int pidx;
		while(trainIds.size() < numLabels) {
			while(sampled[pidx = sampler.nextInt(pool.length)]) ;
			trainIds.add(pool[pidx]);
			sampled[pidx] = true;
			getInstance(pool[pidx]).isLabeled = true;
		}
		
		for(int i = 0; i < numInstances; i++) {
			if(!getInstance(i).isLabeled) {
				unlabeledIds.add(i);
				if(!getInstance(i).isHeldout)
					devIds.add(i);
			}
		}
		
		trains = trainIds.toNativeArray();
		devs = devIds.toNativeArray();
		tests = testIds.toNativeArray();	
		unlabeled = unlabeledIds.toNativeArray();
		
		System.out.println("Number of trains:\t" + trains.length + "\tdevs:\t"
				+ devs.length + "\ttests:\t" + tests.length + "\tunlabeled:\t"
				+ unlabeled.length);
	}
	
	public void resetLabels(int[] newTrains, int[] newDevs, int[] newTests)	{
		trains = new int[newTrains.length];
		devs = new int[newDevs.length];
		tests = new int[newTests.length];
		unlabeled = new int[newDevs.length + newTests.length];
		int nrTrains = 0, nrTests = 0, nrDevs = 0, nrUnlabeled = 0;
		
		for(int sid : newTrains) {
			AbstractSequence instance = getInstance(sid);
			instance.isLabeled = true;
			instance.isHeldout = false;
			trains[nrTrains++] = sid;
		}
		
		for(int sid : newDevs) {
			AbstractSequence instance = getInstance(sid);
			instance.isLabeled = false;
			instance.isHeldout = false;
			devs[nrDevs++] = sid;
			unlabeled[nrUnlabeled++] = sid;
		}
		
		for(int sid : newTests) {
			AbstractSequence instance = getInstance(sid);
			instance.isLabeled = false;
			instance.isHeldout = true;
			tests[nrTests++] = sid;
			unlabeled[nrUnlabeled++] = sid;
		}
		
		Arrays.sort(trains);
		Arrays.sort(devs);
		Arrays.sort(tests);
		Arrays.sort(unlabeled);
		System.out.println(String.format("Reset labels with: trains (%d)," +
				" devs (%d), tests (%d), unlabeled (%d)", nrTrains, nrDevs,
				nrTests, nrUnlabeled));
	}

	public void printCrossValidationInfo() {
		System.out.print(String.format("\ntrains (%d):\t", trains.length));
		for(int i : trains) { 	
			System.out.print(i + " " );
		}
		
		System.out.print(String.format("\ndevs: (%d)\t",  devs.length));
		System.out.print(String.format("\ntests: (%d)\t",  tests.length));
		
		System.out.println();
	}
	
	public String getTag(int tid) {
		return "";
	}
	
	public String getPrintableWord(int wordID) {
		return "";
	}
	
	public String getPrintableTag(int tagID) {
		return "";
	}
}
