package it.unipd.dei.dm1617.project;

import org.apache.spark.api.java.*;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.feature.Word2Vec;
import org.apache.spark.mllib.feature.Word2VecModel;
import org.apache.spark.mllib.linalg.BLAS;


import java.util.ArrayList;

import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;

public class DocumentRepresentationWord2Vec implements DocumentRepresentation {

    private static int vectorSize;
    private static int numPartitions;
    private static int numIterations;
    private static int wordMinCount;
    private static Broadcast<Word2VecModel> broadcast_model;
    private Word2VecModel model;
    private JavaSparkContext sc;

    DocumentRepresentationWord2Vec(JavaSparkContext sc, int inVectorSize, int inNumPartitions, int inNumIterations, int inWordMinCount)
    {
        this.sc = sc;
        vectorSize = inVectorSize;
        numPartitions = inNumPartitions;
        numIterations = inNumIterations;
        wordMinCount = inWordMinCount;
    }

    /**
     * Perform the conversion to Vector of the categories.
     *
     * @param dCategories [Category]
     * @return dCategoryMap: [Category, {Vector,#occurrences}]
     */
    public static JavaPairRDD<String, Tuple2<Vector, Integer>> transformCategories(JavaRDD<TextWikiPage> dCategories) {
        return dCategories.flatMapToPair((page) -> {
            ArrayList<String> categories = page.getCategories();
            ArrayList<Tuple2<String, Vector>> transfCategories = new ArrayList<>();
            for (String cat : categories) {
                try {
                    transfCategories.add(new Tuple2<>(cat, transformWord(cat)));
                } catch (Exception e) {
                }
            }
            return transfCategories.iterator();
        })  //[Category, Vector]
                .mapToPair((catVec) -> new Tuple2<>(catVec, 1)) //[{Category, Vector},1]
                .reduceByKey((x, y) -> x + y)  //[{Category, Vector},#occurrences]
                .mapToPair((tuple) -> new Tuple2<>(tuple._1._1, new Tuple2<>(tuple._1._2, tuple._2))); //[Category, {Vector,#occurrences}]
    }

    public static JavaRDD<VectorWikiPage> transformDataset(JavaRDD<TextWikiPage> dataset) {
        return dataset
                //The document is represented by the average of the vectors of its words
                .map((page) ->
                {
                    ArrayList<String> text = page.getText();

                    Vector sum = Vectors.zeros(vectorSize);

                    int counter = 0;

                    for (String word : text) {
                        //Manage the words that occur less than wordMinCount
                        try {
                            BLAS.axpy(1, transformWord(word), sum);
                            ++counter;
                        } catch (Exception e) {
                        }
                    }

                    BLAS.scal(counter, sum);

                    return new VectorWikiPage(page.getId(), page.getCategories(), sum);
                })
                .filter((page) -> Vectors.norm(page.getVector(), 2) > 0); //Don't consider vectors with zero norm
    }

    static private Vector transformWord(String word) {
        return broadcast_model.getValue().transform(word);
    }

    public Word2VecModel getModel() {
        return model;
    }

    public void setModel(Word2VecModel model) {
        this.model = model;
    }

    public void saveModel(String filename) {
        model.save(JavaSparkContext.toSparkContext(sc), filename);
    }

    public void loadModel(String filename) {
        model = Word2VecModel.load(JavaSparkContext.toSparkContext(sc), filename);
    }

    public void trainModel(JavaRDD<ArrayList<String>> dDataset) {
        Word2Vec support = new Word2Vec();

        support.setMinCount(wordMinCount);
        support.setVectorSize(vectorSize);
        support.setNumIterations(numIterations);
        support.setNumPartitions(numPartitions);
        //TODO SET HYPERPARAMETERS (CROSS-VALIDATION OR VALIDATION?)
        model = support.fit(dDataset);
    }

    public void distributeModel() {
        broadcast_model = sc.broadcast(model);
    }
}
