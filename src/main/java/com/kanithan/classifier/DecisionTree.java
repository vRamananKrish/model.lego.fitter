package com.kanithan.classifier;

import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.IndexToString;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.StringIndexerModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.ml.feature.VectorIndexerModel;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.SQLContext;

public class DecisionTree {

	private StringIndexerModel labelIndexer;
	private VectorIndexerModel featureIndexer;
	
	private VectorAssembler vectorAssembler;
	
	private SQLContext sqlContext;
	private SparkContext sparkContext;

	private DataFrame trainingSet;
	private DataFrame testSet;

	private DataFrame dataSet;

	
	final static Logger logger = Logger.getLogger(DecisionTree.class);
	

	public DecisionTree() {
		
	}
	
	public DecisionTree(String fileName, float train, float test) {
		logger.info("Decision Tree, initialized ---------");
		
		SparkConf config = new SparkConf().setAppName("ModelFitting").setMaster("local");
		
		sparkContext = new SparkContext(config);

		sqlContext = new SQLContext(sparkContext);

		loadDataSet(fileName, train, test);

	}

	/**
	 * Load dataset into DataFrame
	 * 
	 */
	public void loadDataSet(String fileName, float train, float test) {
		logger.info("Loading the data.... -"+ fileName+" with a split:"+ train +" & "+ test);
		
		DataFrame data = sqlContext.read().format("com.databricks.spark.csv").option("header", "true").load(fileName);
		
		data.printSchema();
		
		featureSelection(data);
		
		// Initialize the LabelIndexer
		DataFrame indxedVector = featureSelection(data);
		
		indxedVector.printSchema();
		
		DataFrame[] splits = indxedVector.randomSplit(new double[] { train, test });

		this.trainingSet = splits[0];
		this.testSet = splits[1];

	}
	
	
	/**
	 * Feature selection
	 * 
	 * */
	public DataFrame featureSelection(DataFrame inputData){
		
		
		DataFrame sepalLengthIndx = new StringIndexer().setInputCol("sepallength").setOutputCol("sepallengthIndx").fit(inputData).transform(inputData);
		
		DataFrame sepalWidthIndx = new StringIndexer().setInputCol("sepalwidth").setOutputCol("sepalwidthIndx").fit(sepalLengthIndx).transform(sepalLengthIndx);
		
		DataFrame petalLengthIndx = new StringIndexer().setInputCol("petallength").setOutputCol("petallengthIndx").fit(sepalWidthIndx).transform(sepalWidthIndx);
		
		DataFrame petalWidthIndx = new StringIndexer().setInputCol("petalwidth").setOutputCol("petalwidthIndx").fit(petalLengthIndx).transform(petalLengthIndx);
		
		DataFrame classIndx = new StringIndexer().setInputCol("class").setOutputCol("classIndx").fit(petalWidthIndx).transform(petalWidthIndx);
		
		this.vectorAssembler = new VectorAssembler().setInputCols(new String[]{"sepallengthIndx","sepalwidthIndx","petallengthIndx","petalwidthIndx"}).setOutputCol("features");
		
		return classIndx;
	}
	
	
	/**
	 * Creation of pipeline with transformers and estimators
	 * 
	 */
	public DataFrame createPipeLine() {
		
		logger.info("Pipeline creation method");
		
		// Chain the dataframes, transformers and estimators
		Pipeline pipeline = new Pipeline().setStages(
				new PipelineStage[] {this.vectorAssembler, getDecisionTree(), getTransformer()});

		PipelineModel model = pipeline.fit(this.trainingSet);

		DataFrame predictions = model.transform(this.testSet);

		// Show maximum number of records
		predictions.show(30);
		
		return predictions;

	}

	/**
	 * Get classifier
	 * 
	 */
	private DecisionTreeClassifier getDecisionTree() {

		DecisionTreeClassifier decisionTree = new DecisionTreeClassifier().setLabelCol("classIndx").setMaxDepth(5).setMaxBins(50);

		return decisionTree;

	}
	
	private LogisticRegression getLogisticRegression(){
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
	public double evaluateAndPredict() {
		logger.info("Evaluate and predict method");
		// Create pipeline and get the predictions
		DataFrame predictions = createPipeLine();

		// Compute the test
		MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
				.setLabelCol("classIndx")
				.setPredictionCol("prediction")
				.setMetricName("weightedPrecision");// "recall", "precision", "weightedRecall"
		
		logger.info("Multi class evaluator : "+ evaluator.getPredictionCol());
		
		return (evaluator.evaluate(predictions) * 100);
		
	}

}
