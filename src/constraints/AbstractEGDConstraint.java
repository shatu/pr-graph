package constraints;

public interface AbstractEGDConstraint {

	double getEdgeMarginal(int sid, int i, int s, int sp);
	
	double getNodeMarginal(int sid, int i, int s);

}
