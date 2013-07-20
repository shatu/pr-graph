package analysis;

public class Utils {

	
	public static double entropy(int[] p) {
		double norm = .0, ent = .0;
		for(int px : p) norm += px;
		for(int px : p) 
			if(px > 0) {
				ent -= (1.0 * px / norm) * Math.log(1.0 * px / norm) / Math.log(2.0);
			}
		return ent;
	}
	
	public static double entropy(double[] p) {
		double norm = .0, ent = .0;
		for(double px : p) norm += px;
		for(double px : p) 
			if(px > 0) {
				ent -= (px / norm) * Math.log(px / norm) / Math.log(2.0);
			}
		return ent;
	}
}
