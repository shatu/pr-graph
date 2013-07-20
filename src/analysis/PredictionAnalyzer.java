package analysis;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import models.SecondOrderFactorGraph;
import data.AbstractCorpus;
import data.AbstractSequence;

public class PredictionAnalyzer 
{

	AbstractCorpus corpus;
	
	int[][] pred;
	
	int[] tagFreq;
	int unseenWordFreq, unseenBitagFreq, seenWordFreq, seenBitagFreq;
	double[] tagAcc;
	double unseenWordAcc, unseenBitagAcc, seenWordAcc, seenBitagAcc;
	int[] seenWords;
	int[] seenBitags;
	int tsquared;
	
	public PredictionAnalyzer(AbstractCorpus corpus) {
		this.corpus = corpus;
		pred = new int[corpus.numInstances][];
		for(int i = 0; i < corpus.numInstances; i++) 
			pred[i] = new int[corpus.getInstance(i).length];

		tsquared = corpus.numTags * corpus.numTags;
		
		tagFreq = new int[corpus.numTags];
		tagAcc = new double[corpus.numTags];
		
		// pre-compute unseen words
		seenWords = new int[corpus.numWords];
		Arrays.fill(seenWords, 0);
		// pre-compute unseen tags
		seenBitags = new int[tsquared + corpus.numTags];
		Arrays.fill(seenBitags, 0);
		
		for(int sid : corpus.trains) { 
			int[] tokens = corpus.getInstance(sid).tokens;
			int[] tags = corpus.getInstance(sid).tags;
			for(int i = 0; i < tokens.length; i++) {
				seenWords[tokens[i]] = 1;
				seenBitags[getBitag(tags, i)] = 1;
			}
		}
		
		Arrays.fill(tagFreq, 0);
		unseenWordFreq = unseenBitagFreq = seenWordFreq = seenBitagFreq = 0; // compute #tokens
		
		for(int sid : corpus.unlabeled) {
			int[] tokens = corpus.getInstance(sid).tokens;
			int[] tags = corpus.getInstance(sid).tags;
			for(int i = 0; i < tokens.length; i++) {
				++ tagFreq[tags[i]];
				if(seenWords[tokens[i]] == 0) ++ unseenWordFreq;
				else ++ seenWordFreq;
				
				if(seenBitags[getBitag(tags, i)] == 0) ++ unseenBitagFreq;
				else ++ seenBitagFreq;
			}
		}
		
	}

	protected int getBitag(int[] tags, int idx) {
		return idx > 0 ? (tags[idx-1] * corpus.numTags + tags[idx]) : tsquared + tags[idx];
	}

	public void put(AbstractSequence instance, SecondOrderFactorGraph model)
	{
		int sid = instance.seqID;
		for(int i = 0; i < instance.length; i++)
			pred[sid][i] = model.decode[i];
	}
	
	public void output(String outputFilePath, int[][] predictions) throws IOException
	{
		Arrays.fill(tagAcc, 0.0);
		unseenWordAcc = unseenBitagAcc = seenWordAcc = seenBitagAcc = 0.0;
		
		BufferedWriter fout = new BufferedWriter(new FileWriter(outputFilePath));
		
		for(int sid : corpus.unlabeled) {
			int[] tokens = corpus.getInstance(sid).tokens;
			int[] tags = corpus.getInstance(sid).tags;
			
			fout.write(String.format("%d", sid));
			
			for(int i = 0; i < predictions[sid].length; i++) {
				fout.write("\t" + predictions[sid][i]);
				
				if(predictions[sid][i] == tags[i]) {
					++ tagAcc[tags[i]];
					if(seenWords[tokens[i]] == 0)
						++ unseenWordAcc;
					else 
						++ seenWordAcc;
					
					if(seenBitags[getBitag(tags, i)] == 0)
						++ unseenBitagAcc;
					else
						++ seenBitagAcc;
				}
			}
			fout.write("\n");
		}
		
		fout.close();
		
		System.out.println("Acc on seen word:\t" + (100.0 * seenWordAcc / seenWordFreq) 
				+ "\tunseen:\t" + (100.0 * unseenWordAcc / unseenWordFreq));
		System.out.println("Acc on seen bitag:\t" + (100.0 * seenBitagAcc / seenBitagFreq) 
				+ "\tunseen:\t" + (100.0 * unseenBitagAcc / unseenBitagFreq));
		System.out.println("Error breakdown by tags");
		for(int i = 0; i < corpus.numTags; i++) {
			System.out.println(corpus.getTag(i) + "\t" + (100.0 * tagAcc[i] / tagFreq[i]));
		}
	}
	
	public void output(String outputFilePath) throws IOException
	{
		// output predictions
		Arrays.fill(tagAcc, 0.0);
		unseenWordAcc = unseenBitagAcc = seenWordAcc = seenBitagAcc = 0.0;
		
		BufferedWriter fout = new BufferedWriter(new FileWriter(outputFilePath));
		
		for(int sid : corpus.unlabeled) {
			int[] tokens = corpus.getInstance(sid).tokens;
			int[] tags = corpus.getInstance(sid).tags;
			
			fout.write(String.format("%d", sid));
			
			for(int i = 0; i < pred[sid].length; i++) {
				fout.write("\t" + pred[sid][i]);
				
				if(pred[sid][i] == tags[i]) {
					++ tagAcc[tags[i]];
					if(seenWords[tokens[i]] == 0)
						++ unseenWordAcc;
					else 
						++ seenWordAcc;
					
					if(seenBitags[getBitag(tags, i)] == 0)
						++ unseenBitagAcc;
					else
						++ seenBitagAcc;
				}
			}
			fout.write("\n");
		}
		
		fout.close();
		
		System.out.println("Acc on seen word:\t" + (100.0 * seenWordAcc / seenWordFreq) 
				+ "\tunseen:\t" + (100.0 * unseenWordAcc / unseenWordFreq));
		System.out.println("Acc on seen bitag:\t" + (100.0 * seenBitagAcc / seenBitagFreq) 
				+ "\tunseen:\t" + (100.0 * unseenBitagAcc / unseenBitagFreq));
		System.out.println("Error breakdown by tags");
		for(int i = 0; i < corpus.numTags; i++) {
			System.out.println(corpus.getTag(i) + "\t" + (100.0 * tagAcc[i] / tagFreq[i]));
		}
	}
}
