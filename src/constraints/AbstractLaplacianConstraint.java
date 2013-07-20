package constraints;

public interface AbstractLaplacianConstraint {

	
	public void project(double[] theta, double[] softEmpirical);
	public double getProjector(int nid, int j);

	public double[] getLambda() ;
}
