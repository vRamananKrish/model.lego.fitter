package com.kanithan;

import java.io.Serializable;

import org.junit.After;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;


import com.kanithan.classifier.ModelLegoFitter;

/**
 * Unit test for simple App.
 */
//@Ignore
public class IncidentDataFitterTest implements Serializable {

	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;

	private ModelLegoFitter modelLegoFitter;
	

	@Before
	public void setUp() {
		
		String fileName = this.getClass().getClassLoader().getResource("incidents-v1.0.csv").toString();
		
		System.out.println("File input : "+ fileName);
		
//		String[] columns = {"state","priority","impact","urgency","prev_state","ka_ref","closure_code", "class"};
		
		String[] columns = {"priority","ka_ref","closure_code", "class"};
		
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

		double rfAccuracy = modelLegoFitter.evaluateAndPredict("RandomForestClassifier");

		System.out.println("Precision Accuracy value of Decision Tree : " + dtAccuracy+", Random Forest:"+ rfAccuracy);
	}

}
