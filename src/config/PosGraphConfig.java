package config;

import java.io.PrintStream;

import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;

public class PosGraphConfig extends PosConfig {

	@Option(name = "-wikt-path", usage="")
	public String wiktionaryPath =
		"/home/luheng/Working/pr-graph/data/langs/enwikt-defs-latest-all.tsv";
	
	@Option(name = "-num-neighbors", usage="")
	public int numNeighbors = 60;
	
	@Option(name = "-ngram-size", usage="")
	public int ngramSize = 3;
	
	@Option(name = "-context-size", usage="")
	public int contextSize = 5;

	@Option(name = "-mutual", usage="")
	public boolean mutualKNN = true;

	@Option(name = "-min-sim", usage="")
	public double minSimilarity = 0.01;
	
	public PosGraphConfig(String[] args) {
		super(args);
		CmdLineParser parser = new CmdLineParser(this);
		parser.setUsageWidth(120);
		try {
			parser.parseArgument(args);
		} catch (CmdLineException e) {
			e.printStackTrace();
		}
	}

	public void print(PrintStream ostr) {
		ostr.println("-wikt-path\t" + wiktionaryPath);
		ostr.println("-ngram-size\t" + ngramSize);
		ostr.println("-context-path\t" + contextSize);
		ostr.println("-mutual\t" + mutualKNN);
		ostr.println("-min-sim\t" + minSimilarity);
		super.print(ostr);
	}
}
