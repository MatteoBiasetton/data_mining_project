package it.unipd.dei.dm1617.project;

import it.unipd.dei.dm1617.CountVectorizer;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.feature.IDF;
import org.apache.spark.mllib.feature.IDFModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;
import java.util.ArrayList;

public class BagOfWordsSpark implements DocumentRepresentation {

    private static Broadcast<IDFModel> broadcast_model;
    private static JavaRDD<Vector> tf;
    private int wordMinCount;
    private int vectorSize;
    private IDFModel idfModel;
    private JavaSparkContext sc;

    BagOfWordsSpark(JavaSparkContext sc, int vectorSize, int wordMinCount) {
        this.sc = sc;
        this.vectorSize = vectorSize;
        this.wordMinCount = wordMinCount;
    }

    public static JavaRDD<VectorWikiPage> transformDataset(JavaRDD<TextWikiPage> dataset) {
        JavaRDD<Vector> vectors = broadcast_model.getValue().transform(tf);

        return dataset.zip(vectors).map((page) -> new VectorWikiPage(page._1.getId(), page._1.getCategories(), page._2)).filter((page) -> Vectors.norm(page.getVector(), 2) > 0);
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
                    transfCategories.add(new Tuple2<>(cat, null));
                } catch (Exception e) {
                }
            }
            return transfCategories.iterator();
        })  //[Category, Vector]
                .mapToPair((catVec) -> new Tuple2<>(catVec, 1)) //[{Category, Vector},1]
                .reduceByKey((x, y) -> x + y)  //[{Category, Vector},#occurrences]
                .mapToPair((tuple) -> new Tuple2<>(tuple._1._1, new Tuple2<>(tuple._1._2, tuple._2))); //[Category, {Vector,#occurrences}]
    }

    @Override
    public void saveModel(String filename) {

    }

    @Override
    public void loadModel(String filename) {

    }

    @Override
    public void trainModel(JavaRDD<ArrayList<String>> dDataset) {

        // Transform the sequence of lemmas in vectors of counts in a
        // space of 100 dimensions, using the 100 top lemmas as the vocabulary.
        // This invocation follows a common pattern used in Spark components:
        //
        //  - Build an instance of a configurable object, in this case CountVectorizer.
        //  - Set the parameters of the algorithm implemented by the object
        //  - Invoke the `transform` method on the configured object, yielding
        //  - the transformed dataset.
        //
        // In this case we also cache the dataset because the next step,
        // IDF, will perform two passes over it.
        tf = new CountVectorizer()
                .setVocabularySize(vectorSize)
                .transform(dDataset)
                .cache();

/*
        // Apply transformation ignoring terms that occur in less than a minimum of documents
        HashingTF htf = new HashingTF();
        tf = htf.transform(dDataset).cache();
*/
        idfModel = new IDF(wordMinCount).fit(tf);
    }

    @Override
    public void distributeModel() {
        broadcast_model = sc.broadcast(idfModel);
    }
}
