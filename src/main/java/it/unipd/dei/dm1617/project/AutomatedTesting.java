package it.unipd.dei.dm1617.project;

import it.unimi.dsi.fastutil.objects.Object2ObjectOpenHashMap;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.linalg.Vector;
import scala.Tuple2;

import java.util.ArrayList;

public class AutomatedTesting {
    public static final int COSINE_DISTANCE = 0;
    public static final int EUCLEDIAN_DISTANCE = 1;
    private static final int WORD_2_VEC = 0;
    private static final int BAG_OF_WORDS = 1;
    private static final int MAP_MOST_FREQUENT = 0;
    private static final int MAP_MOST_K_FREQUENT = 1;
    private static final int MAP_NEAREST = 2;
    private static final int CLUSTERING_K_CENTER = 0;
    private static final int CLUSTERING_K_MEANS = 1;
    private static final int CLUSTERING_K_MEDIANS = 2;
    private static final int CLUSTERING_LDA = 3;

    private static Broadcast<Object2ObjectOpenHashMap<String, Tuple2<Vector, Integer>>> bCategoryMap;

    public static void main(String[] args) {

        // Initial Configuration
        String datasetLemmatizedPath = "./data/0.01-unc-msd-lemmatized.dat";

        // Choose test parameters
        int documentRepresentation = WORD_2_VEC;
        int clusteringType = CLUSTERING_K_MEANS;
        int distanceType = EUCLEDIAN_DISTANCE;
        int kFreqCateg = 3;

        int[] clusterNumberInterval = {3, 5, 6, 7};//{10, 50, 100, 200, 500, 1000}; //OK
        int[] wordMinCountInterval = {1, 5, 10}; //OK
        int[] vectorSizeInterval = {100, 300, 500}; //OK
        int[] mappingTypeInterval = {MAP_NEAREST, MAP_MOST_FREQUENT, MAP_MOST_K_FREQUENT}; //OK

        // Initialize Spark
        SparkConf sparkConf = new SparkConf(true).setAppName("Word count optimization");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);
        sc.setLogLevel("ERROR");

        // Compute document representation
        Preprocessing pre = new Preprocessing(sc, distanceType);
        JavaRDD<TextWikiPage> dDataset = pre.loadSplitted(datasetLemmatizedPath).cache();

        for (int wordMinCount : wordMinCountInterval) {
            for (int vectorSize : vectorSizeInterval) {
                for (int mappingType : mappingTypeInterval) {
                    JavaRDD<VectorWikiPage> dVectorPages = documentRepresentation(sc, dDataset, distanceType, mappingType, wordMinCount, documentRepresentation, vectorSize, kFreqCateg);

                    for (int clusterNumber : clusterNumberInterval) {
                        test(sc, dVectorPages, clusteringType, distanceType, mappingType, clusterNumber, vectorSize, wordMinCount);
                    }
                }
            }
        }
    }

    private static JavaRDD<VectorWikiPage> documentRepresentation(JavaSparkContext sc, JavaRDD<TextWikiPage> dDataset, int distanceType, int mappingType, int wordMinCount, int documentRepresentation, int vectorSize, int kFreqCateg) {
        //WORD2VEC parameters
        int numPartitions = 8; //FIXME
        int numIterations = 8; //FIXME

        DocumentRepresentation docRep = null;
        switch (documentRepresentation) {
            case BAG_OF_WORDS:
                docRep = new BagOfWordsSpark(sc, vectorSize, wordMinCount);
                break;
            case WORD_2_VEC:
                docRep = new DocumentRepresentationWord2Vec(sc, vectorSize, numPartitions, numIterations, wordMinCount);
                break;
        }

        docRep.trainModel(dDataset.map(TextWikiPage::getText).cache());
        docRep.distributeModel();

        JavaRDD<VectorWikiPage> dFullVectorPages = null;
        JavaPairRDD<String, Tuple2<Vector, Integer>> dVectorCategory = null;
        switch (documentRepresentation) {
            case BAG_OF_WORDS:
                dFullVectorPages = BagOfWordsSpark.transformDataset(dDataset).cache();
                dVectorCategory = BagOfWordsSpark.transformCategories(dDataset).cache();
                break;
            case WORD_2_VEC:
                dFullVectorPages = DocumentRepresentationWord2Vec.transformDataset(dDataset).cache();
                dVectorCategory = DocumentRepresentationWord2Vec.transformCategories(dDataset).cache();
                break;
        }

        bCategoryMap = sc.broadcast(new Object2ObjectOpenHashMap<>(dVectorCategory.collectAsMap()));

        //Various mapping methods for the categories of a VectorWikiPage
        JavaRDD<VectorWikiPage> dVectorPages = null;
        Preprocessing pre = new Preprocessing(sc, distanceType);
        switch (mappingType) {
            case MAP_MOST_FREQUENT:
                dVectorPages = pre.mapCategoriesToFrequent(dFullVectorPages, bCategoryMap);
                break;
            case MAP_MOST_K_FREQUENT:
                dVectorPages = pre.mapCategoriesToKFrequent(dFullVectorPages, bCategoryMap, kFreqCateg);
                break;
            case MAP_NEAREST:
                dVectorPages = pre.mapCategoriesToNearest(dFullVectorPages, bCategoryMap);
        }

        return dVectorPages;
    }

    private static void test(JavaSparkContext sc, JavaRDD<VectorWikiPage> dVectorPages, int clusteringType, int distanceType, int mappingType, int k, int vectorSize, int wordMinCount) {
        // Perform clustering
        JavaPairRDD<Long, VectorWikiPage> dVectorsClusters = null;
        switch (clusteringType) {
            case CLUSTERING_K_CENTER:
                //System.out.println("K-Center clustering...");
                dVectorsClusters = (new ClusteringKCenters(sc, distanceType)).getClusters(dVectorPages, k);
                break;
            case CLUSTERING_K_MEDIANS:
                //System.out.println("K-Median clustering...");
                dVectorsClusters = (new ClusteringKMedians(sc, distanceType)).getClusters(dVectorPages, k);
                break;
            case CLUSTERING_K_MEANS:
                //System.out.println("K-Means clustering...");
                ClusteringKMeans kMeans = new ClusteringKMeans(sc);
                JavaRDD<Vector> dVectors = dVectorPages.map(VectorWikiPage::getVector);
                kMeans.trainModel(dVectors.rdd().cache(), k, 20);

                kMeans.distributeModel();
                dVectorsClusters = ClusteringKMeans.predict(dVectorPages);
                break;
            case CLUSTERING_LDA:
                //System.out.println("LDA clustering...");
                ClusteringLDA clda = new ClusteringLDA(sc, dVectorPages);
                clda.trainModel(k, 20);
                dVectorsClusters = clda.getClustering();
                break;
        }

        //SUPERVISED TESTS
        SupervisedEvaluation se = new SupervisedEvaluation(sc, distanceType);
        JavaPairRDD<Long, Integer> clustersSize = se.computeClusterSize(dVectorsClusters);
        ArrayList<Tuple2<Long, Integer>> lClusterSize = new ArrayList<>(clustersSize.sortByKey().collect());
        Broadcast<ArrayList<Tuple2<Long, Integer>>> bClusterSize = sc.broadcast(lClusterSize);


        //System.out.println("\nRAND AND JACCARD");
        String[] lRandAndJaccard;
        if (mappingType != MAP_MOST_K_FREQUENT)
            lRandAndJaccard = se.printRandAndJaccard(dVectorsClusters, bClusterSize);
        else
            lRandAndJaccard = new String[2];

        //System.out.println("Rand: "+lRandAndJaccard[0]);
        //System.out.println("Jaccard: "+lRandAndJaccard[1]);

        //System.out.println("\nENTROPY");
        String[] lEntropy = se.printEntropy(dVectorsClusters, bCategoryMap, bClusterSize, k);

        //System.out.println("AvgClusterEntropy: "+lEntropy[0]);
        //System.out.println("AvgCategoryEntropy: "+lEntropy[1]);
        //System.out.println("Cluster-Entropy: "+lEntropy[2]);


        //UNSUPERVISED TESTS
        UnsupervisedEvaluation ue = new UnsupervisedEvaluation(sc, distanceType, k);

        //System.out.println("\nHOPKINS STATISTICS");
        //ue.printHopkins(dVectorPages);

        //System.out.println("\nSEPARATION");
        String[] lSeparation = ue.printSeparation(dVectorsClusters, bClusterSize, k);

        //System.out.println("\nCOHESION");
        String[] lCohesion = ue.printCohesion(dVectorsClusters, bClusterSize);

        //System.out.println("\nSILHOUETTE");
        String[] lSilhouette = ue.printSilhouette(dVectorsClusters, bClusterSize);


        StringBuilder outClusterSize = new StringBuilder();

        for (Tuple2<Long, Integer> e : lClusterSize)
            outClusterSize.append(e._2).append(";");

        outClusterSize = new StringBuilder(outClusterSize.substring(0, outClusterSize.length() - 1));

        System.out.println(k + ";" + wordMinCount + ";" + vectorSize + ";" + mappingType + ";" + lSeparation[0] + ";" + lCohesion[0] + ";" + lSilhouette[0] + ";" + lRandAndJaccard[0]
                + ";" + lRandAndJaccard[1] + ";" + lEntropy[0] + ";" + lEntropy[1] + ";" + outClusterSize + ";" + lCohesion[1] + ";" + lSilhouette[1] + ";"
                + lEntropy[2]);
    }
}
