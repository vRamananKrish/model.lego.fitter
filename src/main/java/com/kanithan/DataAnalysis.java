package com.kanithan;

import com.kanithan.classifier.ModelLegoFitter;

public class DataAnalysis {

	public static void main(String[] args) {

		float trainSplit = 0.8f;
		float testSplit = 0.2f;

		String[] columns = { "priority", "ka_ref", "closure_code", "class" };

		ModelLegoFitter modelLegoFitter = new ModelLegoFitter(args[0], trainSplit, testSplit, columns);

		double dtAccuracy = modelLegoFitter.evaluateAndPredict("DecisionTree");

		double rfAccuracy = modelLegoFitter.evaluateAndPredict("RandomForestClassifier");

		System.out
				.println("Precision Accuracy value of Decision Tree : " + dtAccuracy + ", Random Forest:" + rfAccuracy);
	}

}
