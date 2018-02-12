package it.unipd.dei.dm1617.project;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.clustering.KMeans;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.rdd.RDD;
import scala.Tuple2;

public class ClusteringKMeans
{

    private static Broadcast<KMeansModel> broadcast_model;
    private KMeans kInst;
    private KMeansModel model;
    private JavaSparkContext sc;

    ClusteringKMeans(JavaSparkContext sc) {
        this.sc = sc;
        kInst = new KMeans();
    }

    public static JavaPairRDD<Long, VectorWikiPage> predict(JavaRDD<VectorWikiPage> dVectors) {
        return dVectors
                .mapToPair((page) -> new Tuple2<>((long) broadcast_model.getValue().predict(page.getVector()), page));
    }

    public void saveModel(String filename) {
        model.save(JavaSparkContext.toSparkContext(sc), filename);
    }

    public void loadModel(String filename) {
        model = KMeansModel.load(JavaSparkContext.toSparkContext(sc), filename);
    }

    public void setInitializationSteps(int steps) {
        kInst.setInitializationSteps(steps);
    }

    public void setInitialModel(KMeansModel initialModel) {
        kInst.setInitialModel(initialModel);
    }

    public void setEpsilon(double epsilon) {
        kInst.setEpsilon(epsilon);
    }

    public void setRandomSeed(long seed) {
        kInst.setSeed(seed);
    }

    public KMeansModel trainModel(RDD<Vector> data, int numClusters, int numIterations) {
        model = KMeans.train(data, numClusters, numIterations);
        return model;
    }

    public void distributeModel() {
        broadcast_model = sc.broadcast(model);
    }

    public Vector[] getCenters() {
        return model.clusterCenters();
    }

    public double getObjectiveValue(RDD<Vector> data) {
        return model.computeCost(data);
    }
}
