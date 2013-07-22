package graph;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

public class SuffixDictionary {
	String langName;
	public CountDictionary suffixAlphabet;
	
	public SuffixDictionary(String langName){
		System.out.println("creating suffix dictionary for: " + langName);
		this.langName = langName;
		this.suffixAlphabet = new CountDictionary();
	}
	
	public SuffixDictionary(String langName, String[] suffixes) {
		this(langName);
		for(String suf : suffixes) {
			suffixAlphabet.addOrUpdate(suf);
		}
	}
	
	public void addToDict(String suffix){
		if(suffix.startsWith("-")) {
			suffixAlphabet.addOrUpdate(suffix.substring(1));
		}
		else {
			suffixAlphabet.addOrUpdate(suffix);
		}
	}
	
	public boolean contains(String suf) {
		return suffixAlphabet.str2index.containsKey(suf);
	}
	
	public String[] getAllSuffixes() {
		String[] list = suffixAlphabet.str2index.keys(new String[0]);
		Arrays.sort(list);
		return list;
	}
	
	public String getLongestSuffix(String word) {
		for(int len = word.length()-1; len > 1; len++) {
			String tempSuf = word.substring(word.length() - len, word.length());
			if(contains(tempSuf)) {
				return tempSuf;
			}
		}
		return "";
	}
	
	public String[] getAllApplicableSuffixes(String word) {
		ArrayList<String> sufs = new ArrayList<String>();
		for(int len = 1; len <= word.length(); len++ ) {
			String temp = word.substring(word.length() - len, word.length());
			if(contains(temp)) {
				sufs.add(temp);
			}
		}
		return sufs.toArray(new String[0]);
	}
	
	public static SuffixDictionary fromWiktionaryDumpFile(String fname,
			String langName) throws IOException{
		SuffixDictionary dict = new SuffixDictionary(langName);
		BufferedReader reader = new BufferedReader(new FileReader(fname));
		boolean loaded = false;
		while (reader.ready()) {
			String line = reader.readLine();
			if (!(line.startsWith(langName))) {
				if (!loaded) {
					continue;
				}
				else {
					break;
				}
			}
			loaded = true;
			String[] info = line.split("\t");
			if(!info[2].equalsIgnoreCase("Suffix")) {
				continue;
			}
			assert (info[0].equalsIgnoreCase(langName));
			dict.addToDict(info[1]);
		}
		reader.close();
		
		System.out.println("print extracted suffxies");
		for(String s : dict.suffixAlphabet.index2str) {
			System.out.println(s);
		}
		System.out.println("--------");
		return dict;
	}
	

}
