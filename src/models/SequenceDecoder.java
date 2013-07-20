package models;

public class SequenceDecoder {

	
	public static void decodePosterior(double[][] nodeMar, int[] decoded)
	{
		int length = decoded.length;
		int nrStates = nodeMar[0].length;
		
		for(int i = 0; i < length; i++) {
			decoded[i] = 0;
			for(int j = 1; j < nrStates; j++) 
				if(nodeMar[i][j] > nodeMar[i][decoded[i]])
					decoded[i] = j;
		}
	}
	
	public static void decodeViterbi(double[][] nodeScore, double[][][] edgeScore, int[] decoded)
	{
		int length = decoded.length;
		int numTStates = nodeScore[0].length;
		int numStates = numTStates + 2;
		int initialState = numTStates;
		int finalState = numTStates + 1;
		
		double[][] best = new double[length + 1][numStates];
		int[][] prev = new int[length + 1][numStates];
		
		for(int i = 0; i <= length; i++)
			for(int j = 0; j < numStates; j++) { 
				best[i][j] = i == 0 ? edgeScore[0][j][initialState] + nodeScore[0][j] : Double.NEGATIVE_INFINITY;
				prev[i][j] = i == 0 ? initialState : -1;
			}
		
		for(int i = 1; i < length; i++) 
			for(int j = 0; j < numTStates; j++)  
				for(int k = 0; k < numTStates; k++) {
					double r = best[i-1][k] + edgeScore[i][j][k] + nodeScore[i][j];
					if(r > best[i][j]) {
						best[i][j] = r;
						prev[i][j] = k;
					}
				}
		
		for(int k = 0; k < numTStates; k++) {
			double r = best[length-1][k] + edgeScore[length][finalState][k] + nodeScore[length][finalState];
			if(r > best[length][finalState]) {
				best[length][finalState] = r;
				prev[length][finalState] = k;
			}
		}
			
		decoded[length - 1] = prev[length][finalState];
		for(int i = length - 1; i > 0; i--) {
			decoded[i-1] = prev[i][decoded[i]];
		}
	}
}
