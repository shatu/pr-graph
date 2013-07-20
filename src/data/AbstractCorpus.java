package data;

import gnu.trove.TIntArrayList;

import java.io.PrintStream;
import java.util.Arrays;
import java.util.Random;

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
	
	public void jackknife(int numTrainFolds, int numDevFolds, int numTestFolds)
	{
		TIntArrayList trainIDs = new TIntArrayList();
		TIntArrayList devIDs = new TIntArrayList();
		TIntArrayList testIDs = new TIntArrayList();
		
		for(int i = 0; i < numInstances; i++) {
			AbstractSequence instance = getInstance(i);
			if(instance.foldID < numTrainFolds) {
				trainIDs.add(instance.seqID);
				instance.isLabeled = true;
			}
			else if(instance.foldID < numTrainFolds + numDevFolds) 
				devIDs.add(instance.seqID);
			else if(instance.foldID < numTrainFolds + numDevFolds + numTestFolds)
				testIDs.add(instance.seqID);
		}
		
		trains = trainIDs.toNativeArray();
		Arrays.sort(trains);
		
		if(devIDs.size() > 0) {
			devs = devIDs.toNativeArray();
			Arrays.sort(devs);
		}
		
		if(testIDs.size() > 0) {
			tests = testIDs.toNativeArray();
			Arrays.sort(tests);
		}
	}
	
	public void jackknifeRandom(int numTrains, int numDevs, Random sampler)
	{
		TIntArrayList trainIDs = new TIntArrayList();
		TIntArrayList devIDs = new TIntArrayList();
		TIntArrayList testIDs = new TIntArrayList();
		
		int tid;
		boolean[] sampled = new boolean[numInstances];
		for(int i = 0; i < sampled.length; i++) {
			sampled[i] = false;
		}
		
		while(trainIDs.size() < numTrains) {
			while(sampled[tid = sampler.nextInt(numInstances)]) ;
			trainIDs.add(tid);
			getInstance(tid).isLabeled = true;
			sampled[tid] = true;
		}
		
		if(numDevs > 0) {
			while(devIDs.size() < numDevs) {
				while(sampled[tid = sampler.nextInt(numInstances)]) ;
				devIDs.add(tid);
				sampled[tid] = true;
			}	
		}
		
		for(int i = 0; i < numInstances; i++)
			if(!sampled[i]) testIDs.add(i);
		
		trains = trainIDs.toNativeArray();
		Arrays.sort(trains);
		
		if(devIDs.size() > 0) {
			devs = devIDs.toNativeArray();
			Arrays.sort(devs);
		}
		else devs = new int[0];
		
		if(testIDs.size() > 0) {
			tests = testIDs.toNativeArray();
			Arrays.sort(tests);
		}
		else tests = new int[0];

		System.out.println("num trains: " + trains.length + "\tnum devs: " + devs.length + "\tnum tests: " + tests.length);
	}
	

	public void sampleFromFolder(int numLabels, int seedFolder, int holdoutFolder, Random sampler)
	{
		sampleFromFolderUnsorted(numLabels, seedFolder, holdoutFolder, sampler);
		Arrays.sort(trains);
		Arrays.sort(devs);
		Arrays.sort(tests);
		Arrays.sort(unlabeled);
	}
	
	public void sampleFromFolderUnsorted(int numLabels, int seedFolder, int holdoutFolder, Random sampler)
	{
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
		
		System.out.println("Number of trains:\t" + trains.length + "\tdevs:\t" + devs.length + "\ttests:\t" + 
				tests.length + "\tunlabeled:\t" + unlabeled.length);
	}
	
	public void resetLabels(int[] newTrains, int[] newDevs, int[] newTests)
	{
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
		System.out.println(String.format(
				"Reset labels with: trains (%d), devs (%d), tests (%d), unlabeled (%d)", 
					nrTrains, nrDevs, nrTests, nrUnlabeled));
	}

	public void printCrossValidationInfo()
	{
		
		System.out.print(String.format("\ntrains (%d):\t", trains.length));
		for(int i : trains) 	
			System.out.print(i + " " );
		
		System.out.print(String.format("\ndevs: (%d)\t",  devs.length));
		/*
		for(int i : devs) 	
			System.out.print(i + " ");
		*/
		System.out.print(String.format("\ntests: (%d)\t",  tests.length));
		/*
		System.out.print("\ntests:\t");
		for(int i : tests) 	
			System.out.print(i + " "); */
		
		System.out.println();
	}
	
	public String getTag(int tid)
	{
		return "";
	}
	
	public String getPrintableWord(int wordID) {
		return "";
	}
	
	public String getPrintableTag(int tagID) {
		return "";
	}
	
}
