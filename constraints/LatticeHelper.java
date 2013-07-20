package constraints;

import java.util.Random;

public class LatticeHelper {


	public static void deepFill(double[] arr, int filler) {
		for(int i = 0; i < arr.length; i++)
			arr[i] = filler;
	}
		
	public static void deepFill(double[][] arr, double filler)
	{
		for(int i = 0; i < arr.length; i++)
			for(int j = 0; j < arr[i].length; j++)
				arr[i][j] = filler;
	}
	
	public static void deepFill(double[][][] arr, double filler)
	{
		for(int i = 0; i < arr.length; i++)
			for(int j = 0; j < arr[i].length; j++)
				for(int k = 0; k < arr[i][j].length; k++) arr[i][j][k] = filler;
	}
	
	public static void deepCopy(double[][] src, double[][] dest)
	{
		for(int i = 0; i < src.length; i++)
			for(int j = 0; j < src[i].length; j++)
				dest[i][j] = src[i][j];
	}
	
	public static void deepCopy(double[][][] src, double[][][] dest)
	{
		for(int i = 0; i < src.length; i++)
			for(int j = 0; j < src[i].length; j++)
				for(int k = 0; k < src[i][j].length; k++) 
					dest[i][j][k] = src[i][j][k];
	}
	
	public static int getMaxIndex(double[] p) {
		int maxi = 0;
		for(int i = 1; i < p.length; i++)
			if(p[maxi] < p[i]) maxi = i;
		return maxi;
	}
	
	
	public static double logsum(double loga, double logb)
	{		
		if(Double.isInfinite(loga))
			return logb;
		if(Double.isInfinite(logb))
			return loga;

		if(loga > logb) 
			return Math.log1p(Math.exp(logb - loga)) + loga;
		else 
			return Math.log1p(Math.exp(loga - logb)) + logb; 
	}
	
	/*
	 * in place log sum
	public static double logsum(double[] tosum, int length)
	{	
		for(int i = 1; i < length; i += i) {
			for(int j = 0, k = i; k < length; ) {
				if(!Double.isInfinite(tosum[k])) {
					if(Double.isInfinite(tosum[j]))
						tosum[j] = tosum[k];
					else if(tosum[j] > tosum[k])
						tosum[j] = Math.log1p(Math.exp(tosum[k] - tosum[j])) + tosum[j];
					else 
						tosum[j] = Math.log1p(Math.exp(tosum[j] - tosum[k])) + tosum[k];
				}
				j = k + 1;
				k = j + i;
			}
		}
		return tosum[0];
	}
	*/
	
	/*
	 * external log sum
	 */
	public static double logsum(double[] tosum, int length)
	{	
		if(length == 1) return tosum[0];
					
		int idx = 0;
		for(int i = 1; i < length; i++)
			if(tosum[i] > tosum[idx]) idx = i;
		
		double maxx = tosum[idx];
		double sumexp = 0;
		
		for(int i = 0; i < length; i++)
			if(i != idx) sumexp += Math.exp(tosum[i] - maxx);
		
		return Math.log1p(sumexp) + maxx;
	}
	
	public static void main(String[] args)
	{
		int niters = 1000000;
		double[] arr = new double[50];
		double[] arr2 = new double[50];
		double res1 = 0, res2 = 0, res3 = 0;
		
		// test log summer ...
		Random sampler = new Random(12345);
		StopWatch timer = new StopWatch();
		timer.start();
		for(int t = 0; t < niters; t++) {
			for(int i = 0; i < 50; i++)
				arr[i] = sampler.nextGaussian() * 100;
				double ls = Double.NEGATIVE_INFINITY;
				for(int i = 0; i < 50; i++)
					ls = logsum(ls , arr[i]);
				res1 += ls;
		}
		timer.stop();
		System.out.println("log summer 1\t" + res1 + "\t" + timer.getElapsedTime());
		/*
		sampler = new Random(12345);
		timer = new StopWatch();
		timer.start();
		for(int t = 0; t < niters; t++) {
			for(int i = 0; i < 50; i++)
				arr[i] = sampler.nextGaussian() * 100;
				double ls = logsum2(arr, 50);
				res2 += ls;
		}
		timer.stop();
		System.out.println("log summer 2\t" + res2 + "\t" + timer.getElapsedTime());
		*/
		sampler = new Random(12345);
		timer = new StopWatch();
		timer.start();
		for(int t = 0; t < niters; t++) {
			for(int i = 0; i < 50; i++)
				arr[i] = sampler.nextGaussian() * 100;
				double ls = logsum(arr, 50);
				res3 += ls;
		}
		
		System.out.println("log summer 3\t" + res3 + "\t" + timer.getElapsedTime());
	}

}

/*
Copyright (c) 2005, Corey Goldberg

StopWatch.java is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.
*/

class StopWatch 
{    
    private long startTime = 0;
    private long stopTime = 0;
    private boolean running = false;
   
    public void start() {
        this.startTime = System.currentTimeMillis();
        this.running = true;
    }

    public void stop() {
        this.stopTime = System.currentTimeMillis();
        this.running = false;
    }    
    //elaspsed time in milliseconds
    public long getElapsedTime() {
        long elapsed;
        if (running) {
             elapsed = (System.currentTimeMillis() - startTime);
        }
        else {
            elapsed = (stopTime - startTime);
        }
        return elapsed;
    }
   
    //elaspsed time in seconds
    public long getElapsedTimeSecs() {
        long elapsed;
        if (running) {
            elapsed = ((System.currentTimeMillis() - startTime) / 1000);
        }
        else {
            elapsed = ((stopTime - startTime) / 1000);
        }
        return elapsed;
    }

}
