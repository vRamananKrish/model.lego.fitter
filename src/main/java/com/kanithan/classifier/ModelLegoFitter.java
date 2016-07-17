package com.kanithan.classifier;

import java.util.ArrayList;
import java.util.List;

import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.evaluation.Evaluator;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.evaluation.RegressionEvaluator;
import org.apache.spark.ml.feature.IndexToString;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.feature.VectorIndexerModel;
import org.apache.spark.ml.regression.RandomForestRegressor;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.SQLContext;

public class ModelLegoFitter {

	private StringIndexerModel labelIndexer;
	private VectorIndexerModel featureIndexer;

	private VectorAssembler vectorAssembler;

	private SQLContext sqlContext;
	private SparkContext sparkContext;

	private DataFrame trainingSet;
	private DataFrame testSet;

	private String[] columns;
	
	private DataFrame dataSet;

	final static Logger logger = Logger.getLogger(ModelLegoFitter.class);

	public ModelLegoFitter() {

	}

	public ModelLegoFitter(String fileName, float train, float test, String...columns) {

		// assigning the columns
		this.columns = columns;
		
		// Initialize the ModelLegoFitter
		init(fileName, train, test);

	}

	private void init(String dataFileName, float train, float test) {
		logger.info("Decision Tree, initialized ---------");

		SparkConf config = new SparkConf().setAppName("ModelLegoFitter").setMaster("local");

		sparkContext = new SparkContext(config);

		sqlContext = new SQLContext(sparkContext);

		loadDataSet(dataFileName, train, test);

	}

	/**
	 * Load dataset into DataFrame
	 * 
	 */
	public void loadDataSet(String fileName, float train, float test) {
		logger.info("Loading the data.... -" + fileName + " with a split:" + train + " & " + test);

		DataFrame data = sqlContext.read().format("com.databricks.spark.csv").option("header", "true").load(fileName);

		data.printSchema();

		featureSelection(data);

		// Initialize the LabelIndexer
		DataFrame indxedVector = featureIndexer(data, this.columns);

		indxedVector.printSchema();

		DataFrame[] splits = indxedVector.randomSplit(new double[] { train, test });

		this.trainingSet = splits[0];
		this.testSet = splits[1];

	}

	/**
	 * Feature selection
	 * 
	 */
	public DataFrame featureSelection(DataFrame inputData) {

		DataFrame sepalLengthIndx = new StringIndexer().setInputCol("sepallength").setOutputCol("sepallengthIndx")
				.fit(inputData).transform(inputData);

		DataFrame sepalWidthIndx = new StringIndexer().setInputCol("sepalwidth").setOutputCol("sepalwidthIndx")
				.fit(sepalLengthIndx).transform(sepalLengthIndx);

		DataFrame petalLengthIndx = new StringIndexer().setInputCol("petallength").setOutputCol("petallengthIndx")
				.fit(sepalWidthIndx).transform(sepalWidthIndx);

		DataFrame petalWidthIndx = new StringIndexer().setInputCol("petalwidth").setOutputCol("petalwidthIndx")
				.fit(petalLengthIndx).transform(petalLengthIndx);

		DataFrame classIndx = new StringIndexer().setInputCol("class").setOutputCol("classIndx").fit(petalWidthIndx)
				.transform(petalWidthIndx);

		this.vectorAssembler = new VectorAssembler()
				.setInputCols(new String[] { "sepallengthIndx", "sepalwidthIndx", "petallengthIndx", "petalwidthIndx" })
				.setOutputCol("features");

		return classIndx;
	}
	
	/**
	 * Feature selection indexer based on the given column name
	 * 
	 * */
	public DataFrame featureIndexer(DataFrame inputData, String...columns){
		
		DataFrame indexedFrame = inputData;
		List<String> indexCols = new ArrayList<String>();
		
		
		for(String column: columns){
			indexedFrame = new StringIndexer().setInputCol(column).setOutputCol(column+"Indx").fit(indexedFrame).transform(indexedFrame);
			indexCols.add(column+"Indx");
			
		}
		
		String[] indexArr = new String[indexCols.size()];
		indexArr = indexCols.toArray(indexArr);
		
		this.vectorAssembler = new VectorAssembler()
				.setInputCols(indexArr)
				.setOutputCol("features");
		
		return indexedFrame;
		
	}

	/**
	 * Creation of pipeline with transformers and estimators
	 * 
	 */
	public DataFrame createPipeLine(String modelName) {

		logger.info("Pipeline creation method");

		Pipeline pipeline = new Pipeline();
		// Chain the dataframes, transformers and estimators
		if (modelName.equals("RandomForest")) {
			pipeline.setStages(new PipelineStage[] { this.vectorAssembler, getDecisionTree(), getTransformer() });
		} else {
			pipeline.setStages(new PipelineStage[] { this.vectorAssembler, getRandomForestRegressor(), getTransformer() });

		}

		PipelineModel model = pipeline.fit(this.trainingSet);

		DataFrame predictions = model.transform(this.testSet);

		// Show maximum number of records
		predictions.show(40);

		return predictions;

	}

	/**
	 * Get classifier
	 * 
	 */
	private DecisionTreeClassifier getDecisionTree() {

		DecisionTreeClassifier decisionTree = new DecisionTreeClassifier().setLabelCol("classIndx").setMaxDepth(5)
				.setMaxBins(50);

		return decisionTree;

	}
	
	 /** Get classifier
	 * 
	 */
	private RandomForestRegressor getRandomForestRegressor() {

		RandomForestRegressor randomForestRegressor = new RandomForestRegressor().setLabelCol("classIndx").setMaxBins(186);
		

		return randomForestRegressor;

	}

	private LogisticRegression getLogisticRegression() {
		LogisticRegression lr = new LogisticRegression().setLabelCol("classIndx");
		return lr;
	}

	/**
	 * Get transformer
	 * 
	 */
	private IndexToString getTransformer() {

		// Transformer object
		IndexToString labelConverter = new IndexToString().setInputCol("classIndx").setOutputCol("predictedClass");

		return labelConverter;

	}

	/**
	 * Evaluate and predict
	 * 
	 * @return double accuracy
	 */
	public double evaluateAndPredict(String modelName) {
		logger.info("Evaluate and predict method");
		// Create pipeline and get the predictions
		DataFrame predictions = createPipeLine(modelName);
		
		Evaluator evaluator;
		
		if(modelName.equals("RandomForestRegressor")){
			 evaluator = new RegressionEvaluator().setLabelCol("classIndx").setPredictionCol("prediction").setMetricName("rmse");
		}
		else{
			evaluator = new MulticlassClassificationEvaluator().setLabelCol("classIndx")
					.setPredictionCol("prediction").setMetricName("weightedPrecision");// "recall",
																						// "precision",
																						// "weightedRecall"
		}
		
		return (evaluator.evaluate(predictions) * 100);

	}

}
