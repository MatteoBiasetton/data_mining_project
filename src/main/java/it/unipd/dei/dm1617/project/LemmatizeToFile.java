package it.unipd.dei.dm1617.project;

import it.unipd.dei.dm1617.InputOutput;
import it.unipd.dei.dm1617.Lemmatizer;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;

public class LemmatizeToFile {
    public static void main(String[] args) {

        String datasetPath = "./data/little-sample.dat";
        String outputPath = "./data/little-sample-lemmatized.dat";

        // Initialize Spark
        SparkConf sparkConf = new SparkConf(true).setAppName("Word count optimization");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        InputOutput.write(Lemmatizer.lemmatizeWikiPages(InputOutput.read(sc, datasetPath)), outputPath);
    }
}
