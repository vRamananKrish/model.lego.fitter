package com.kanithan;

import com.kanithan.classifier.DecisionTree;

public class DataAnalysis {
	
	public static void main(String[] args) {
		
		float trainSplit = 0.8f;
		float testSplit = 0.2f;
		
		DecisionTree decisionTree = new DecisionTree(args[0], trainSplit, testSplit);
		
		double accuracy = decisionTree.evaluateAndPredict();
		
		System.out.println("Precision Accuracy value : "+ accuracy);
	}
	
	
}
