package config;

import java.io.PrintStream;

import org.kohsuke.args4j.CmdLineException;
import org.kohsuke.args4j.CmdLineParser;
import org.kohsuke.args4j.Option;

public class PosConfig extends Config {
	@Option(name = "-ngram-path", usage="")
	public String ngramPath = "./data/graph/es-60nn-dep.idx";
	
	@Option(name = "-umap-path", usage="")
	public String umapPath = "./data/univmap/es-cast3lb.map";
	
	@Option(name = "-lang-name", usage="")
	public String langName = "spanish";
	
	public PosConfig(String[] args)
	{
		super();
		CmdLineParser parser = new CmdLineParser(this);
		parser.setUsageWidth(120);
		
		try {
			parser.parseArgument(args);
		} catch (CmdLineException e) {
			e.printStackTrace();
		}
	}
	
	public void print(PrintStream ostr)	{
		ostr.println("-ngram-path\t" + ngramPath);
		ostr.println("-umap-path\t" + umapPath);
		ostr.println("-lang-name\t" + langName);
		super.print(ostr);
	}
}
