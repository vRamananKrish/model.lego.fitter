package com.kanithan;

import java.io.Serializable;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import com.kanithan.classifier.ModelLegoFitter;

/**
 * Unit test for simple App.
 */
public class ModelLegoFitterTest implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	private ModelLegoFitter modelLegoFitter;
	

	@Before
	public void setUp() {
		
		String fileName = this.getClass().getClassLoader().getResource("iris.csv").toString();
		
		System.out.println("File input : "+ fileName);
		
		String[] columns = {"sepallength", "sepalwidth", "petallength", "petalwidth", "class"};
		
		modelLegoFitter = new ModelLegoFitter(fileName, 0.8f, 0.2f, columns);
	}

	@After
	public void tearDown() {
		System.out.println("Tear down");
	}

	@Test
	public void testEvaluate() {

//		double accuracy = decisionTree.evaluateAndPredict();
		
		double dtAccuracy = modelLegoFitter.evaluateAndPredict("DecisionTree");

		double rfAccuracy = modelLegoFitter.evaluateAndPredict("RandomForest");

		System.out.println("Precision Accuracy value of Decision Tree : " + dtAccuracy+", Random Forest:"+ rfAccuracy);
	}

}
