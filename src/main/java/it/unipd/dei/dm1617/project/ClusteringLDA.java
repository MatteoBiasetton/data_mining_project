package it.unipd.dei.dm1617.project;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.clustering.DistributedLDAModel;
import org.apache.spark.mllib.clustering.LDA;
import org.apache.spark.mllib.clustering.LDAModel;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.rdd.RDD;
import scala.Tuple2;

public class ClusteringLDA {

    private static Broadcast<LDAModel> broadcast_model;
    LDA lda;
    DistributedLDAModel model;
    JavaRDD<VectorWikiPage> documents;
    private JavaSparkContext sc;
    private int numClusters;

    ClusteringLDA(JavaSparkContext sc, JavaRDD<VectorWikiPage> documents) {
        this.sc = sc;
        this.documents = documents;
        lda = new LDA();
    }

    public DistributedLDAModel trainModel(int numClusters, int maxIterations) {
        this.numClusters = numClusters;
        lda.setMaxIterations(maxIterations);
        model = (DistributedLDAModel) lda.setK(numClusters).run(documents.mapToPair(doc -> new Tuple2<Long, Vector>(doc.getId(), doc.getVector())));
        //clusters the documents (VectorWikipages are mapped into JavaPairRdd<Long,Vector>) into numclusters topics, using LDA.
        return model;
    }

    public JavaPairRDD<Long, VectorWikiPage> getClustering() {
        JavaPairRDD<Long, Vector> topicDist = model.javaTopicDistributions(); //for each document, gets the distribution over all the topics
        //System.out.println("distribution over topics: " + topicDist.count());
        JavaPairRDD<Long, Long> bestTopics = topicDist.mapToPair(doc -> new Tuple2<>((long) doc._2.argmax(), doc._1)); //[ID_CLUSTER, ID_DOC] maps every document to its most likely topic

        bestTopics = bestTopics
                .map((pair) -> pair._1)
                .distinct().zipWithIndex() //[ID_CLUSTER_OLD, ID_CLUSTER_NEW]
                .join(bestTopics) //[ID_CLUSTER_OLD,[ID_CLUSTER_NEW,ID_DOC]
                .mapToPair((joinOut) -> joinOut._2.swap()); //[ID_CLUSTER_NEW,ID_DOC]

        JavaPairRDD<Long, VectorWikiPage> docWithId = documents.mapToPair(v -> new Tuple2<>(v.getId(), v));// create pair (documentId, VectorWikiPage)
        JavaPairRDD<Long, VectorWikiPage> clusters = bestTopics.join(docWithId).mapToPair((joinOut) -> joinOut._2);
        return clusters;
    }

    public void saveModel(String filename) {
        model.save(JavaSparkContext.toSparkContext(sc), filename);
    }

    public void loadModel(String filename) {
        model = DistributedLDAModel.load(JavaSparkContext.toSparkContext(sc), filename);
    }

    public double getObjectiveValue(RDD<Vector> data) {
        return model.logLikelihood();
    }
}