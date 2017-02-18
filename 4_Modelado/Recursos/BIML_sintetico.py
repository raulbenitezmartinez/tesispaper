package Clases;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.FileReader;
import java.io.File;
import java.io.FileWriter;
import java.io.BufferedWriter;
import java.io.StringWriter;
import java.io.PrintWriter;
import java.util.List;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.HashMap;
import java.util.Map;
import java.util.LinkedHashMap;
import java.util.Set;
import java.util.Random;
import java.text.DecimalFormat;
import weka.core.converters.CSVLoader;
import weka.core.converters.ArffSaver;
import weka.core.Attribute;
import weka.core.Instances;//Clase para los ejemplos, tambien llamados instancias.
import weka.core.Instance;
import weka.core.FastVector;//Clase para agrupar las predicciones de la clase Evaluation.
import weka.classifiers.Classifier;//Clase para conjunto de clasificadores.
import weka.classifiers.Evaluation;//Clase para evaluar los clasificadores.
import weka.classifiers.evaluation.NominalPrediction;
import weka.core.SerializationHelper;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;

//BAYES.
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.bayes.NaiveBayesMultinomial;
import weka.classifiers.bayes.NaiveBayesMultinomialUpdateable;
import weka.classifiers.bayes.NaiveBayesUpdateable;

//FUNCTIONS.
import weka.classifiers.functions.Logistic;
import weka.classifiers.functions.MultilayerPerceptron;
import weka.classifiers.functions.SimpleLogistic;
import weka.classifiers.functions.SMO;

//RULES.
import weka.classifiers.rules.DecisionTable;
import weka.classifiers.rules.JRip;
import weka.classifiers.rules.OneR;
import weka.classifiers.rules.PART;
import weka.classifiers.rules.ZeroR;

//TREES.
import weka.classifiers.trees.DecisionStump;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.LMT;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree;
import weka.classifiers.trees.REPTree;

//weka.jar se refiere a Weka 3.8

/** Un programa Java simple.
  * Que hace?.
  * @author Raul Benitez
  * @version 1
  */
public class BIML {
	
	private String filepath = "./Generados/";
    private String filename = "";
    private Instances instancias = null;
	private	Instances train = null;
	private	Instances test = null;
	//private	Instances test1 = null;
	//private	Instances test2 = null;
	private	int trainSize = 0;
	private	int testSize = 0;
	//private	int test1Size = 0;
	//private	int test2Size = 0;
	private long seed = 1; //Default = 1
	private int folds = 10;
	private int max_instancias = 151;
	private double porcentaje_split = 0.8;
	private List<String> lista_claves = null;
	private List<String> etiquetas = null;
	private String cabecera = "";
	private String extra_cabecera = "";
	private String extra_datos = "";
	private FileWriter fstream;
	private BufferedWriter out;
    
    public BIML(String fn, List<String> e, int max_inst){
		
		this.filepath = this.filepath + fn;
		this.filename = fn;
		this.etiquetas = e;
		this.max_instancias = max_inst;
		
		//Se instancian los ejemplos.
		instanciasDesdeFichero();
    }
	
	/**
	  * Loads results from a set of instances contained in the supplied
	  * file.
	  */
    private void instanciasDesdeFichero(){
		
		try{
			
			File f = new File(this.filepath);
			Instances myInsts = null;
			Resample r = null;
			double size_sample = 0;
			
			//Si se trata de un archivo .arff
			if (f.getName().toLowerCase().endsWith(Instances.FILE_EXTENSION)){
				
				//Se lee el archivo ARFF.
				BufferedReader datos = new BufferedReader(new FileReader(f));
				
				//Se instancian los ejemplos.
				//this.instancias = new Instances(datos);
				
				//Resample.
				myInsts = new Instances(datos);

				//
				datos.close();
				
			//Si se trata de un archivo .csv	
			}else if(f.getName().toLowerCase().endsWith(CSVLoader.FILE_EXTENSION)){				
				
				//
				CSVLoader loader = new CSVLoader();
				loader.setSource(f);
				//this.instancias = loader.getDataSet();
				
				//Resample.
				myInsts = loader.getDataSet();
				
			}else{
				
				throw new Exception("El tipo de archivo debe ser .arff o .csv!");
			}
			
			r = new Resample(); 
			r.setNoReplacement(false);
			
			System.out.println("NUM INSTANCES: " + myInsts.numInstances());
				
			size_sample = (double) Math.round(((double)(this.max_instancias * 100) / (double) myInsts.numInstances()) * 100) / 100;
			//size_sample = Double.parseDouble(String.format("%.2f", (double)((double)(this.max_instancias * 100) / (double) myInsts.numInstances())));
			
			System.out.println("SIZE SAMPLE: " + size_sample);
			
			myInsts.setClassIndex(myInsts.numAttributes() - 1);
				
			r.setSampleSizePercent(size_sample); // or whatever % you require 
			r.setInputFormat(myInsts);
			this.instancias = Filter.useFilter(myInsts, r); 

			//Se ramdomizan las filas de ejemplos.
			this.instancias.randomize(new Random(seed));
			
			System.out.println("INSTANCIAS!!!!!!!!!!!!!!: " + this.instancias.numInstances());
			//System.out.println(this.instancias);
			
		}catch(Exception e){
			System.err.println("Este es el problema: " + e);
			e.printStackTrace();
		}
    }
	
	
	/** Se obtienen los conjuntos de entrenamiento y de testeo, a partir del conjunto de instancias.
	  * @param randomizar Entero que puede ser 0 o 1. Un 1 indica que hay que randomizar las ubicaciones.
	  * @param porc_split Double que debe ser mayor a 0 y menor 1. Indica qué porcentaje de las instancias es para ENTRENAMIENTO.
	  * @return No devuelve ningun valor.
	  * @throws No dispara ninguna excepcion.
	  */
    public void trainTestInstancias(int randomizar, double porc_split){
		
		try{
			
			//Se ramdomiza la ubicacion de las instancias.
			if (randomizar == 1){
				
				this.instancias.randomize(new Random(this.seed));				
			}

			//Se realiza el split, se obtienen dos subconjunto: train y test.
			if (porc_split > 0 && porc_split < 1){
				
				this.trainSize = (int) Math.round(this.instancias.numInstances() * porc_split);				
			}else{
				
				this.trainSize = (int) Math.round(this.instancias.numInstances() * this.porcentaje_split);
			}
			
			//
			this.testSize = this.instancias.numInstances() - this.trainSize;
			this.train = new Instances(this.instancias, 0, this.trainSize);
			this.test = new Instances(this.instancias, this.trainSize, this.testSize);
			
			//this.test1 = new Instances(this.instancias, this.trainSize, this.extraSize);
			//this.test2 = new Instances(this.instancias, this.trainSize, this.extraSize);
			
			//Se configura el atributo class.
			this.instancias.setClassIndex(this.instancias.numAttributes() - 1);
			this.train.setClassIndex(this.train.numAttributes() - 1);
			this.test.setClassIndex(this.test.numAttributes() - 1);
					
		}catch(Exception e){
			System.err.println("Error en: " + this.filename);
			System.err.println("Exception: " + e);
			//e.printStackTrace();
		}
    }
	
	/** Se obtienen los conjuntos de entrenamiento y de testeo, a partir del conjunto de instancias.
	  * @param randomizar Entero que puede ser 0 o 1. Un 1 indica que hay que randomizar las ubicaciones.
	  * @param porc_split Double que debe ser mayor a 0 y menor 1. Indica qué porcentaje de las instancias es para entrenamiento.
	  * @return No devuelve ningun valor.
	  * @throws No dispara ninguna excepcion.
	  */
    public void construirYEvaluar(String tipoEvaluacion, int k){
		
			//Conjunto de clasificadores.
			Classifier[] modelos = {
				//BAYES.			
				new BayesNet(),
				new NaiveBayes(),
				new NaiveBayesUpdateable(),
				//FUNCTIONS.			
				new Logistic(),
				new MultilayerPerceptron(),
				new SimpleLogistic(),
				new SMO(),
				//RULES.
				new OneR(),
				new DecisionTable(),
				new JRip(),
				new PART(),
				new ZeroR(),
				//TREES.			
				new DecisionStump(),
				new J48(),
				new LMT(),
				new RandomForest(),
				new RandomTree(),
				new REPTree(),
			};
			
			//Por cada modelo, se construye su clasificador y se evalua.
			for (int j=0; j<modelos.length; j++){
				
				//Se evalua el modelo.
				if (tipoEvaluacion == "split"){
					
					modelo.buildClassifier(this.train);
					evaluar = new Evaluation(this.test);
					evaluar.evaluateModel(modelo, this.test);
					
				}else if (tipoEvaluacion == "crossEstratificado"){
					
					
					modelo.buildClassifier(this.instancias);
					evaluar = new Evaluation(this.instancias);
					evaluar.crossValidateModel(modelo, this.instancias, this.folds, new Random(this.seed));
				}
				
				//INDICADORES RESUMEN:
				
				//Aciertos.
				cantidad_aciertos = evaluar.correct();
				porcentaje_aciertos = evaluar.pctCorrect();
				
				//Kappa.
				kappa_statistic = evaluar.kappa();			
				
				//INDICADORES DE PRECISION POR CLASE:
				for (int q=0; q < this.etiquetas.size(); q++){
					
					//Area under ROC.
					roc_area = evaluar.areaUnderROC(indice_clase);
						
					//Recall.
					recall = evaluar.recall(indice_clase);
						
					//Precision.
					precision = evaluar.precision(indice_clase);
					
					//F-Measure.
					f_measure = evaluar.fMeasure(indice_clase);
						
					//Area under precision-recall curve (AUPRC).
					prc_area = evaluar.areaUnderPRC(indice_clase);	
				}
			
				//MATRIZ DE CONFUSION.
				confusion_matrix = evaluar.confusionMatrix();
			}
    }//Hasta aqui construirYEvaluar.
	
	

	
}//Cierre de la clase BIML.