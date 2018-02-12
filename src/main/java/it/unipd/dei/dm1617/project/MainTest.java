package it.unipd.dei.dm1617.project;

import it.unimi.dsi.fastutil.objects.Object2ObjectOpenHashMap;
import it.unipd.dei.dm1617.Lemmatizer;
import it.unipd.dei.dm1617.WikiPage;
import org.apache.hadoop.yarn.webapp.hamlet.Hamlet;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.linalg.Vector;
import scala.Tuple2;

import java.util.ArrayList;

public class MainTest {
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

    public static void main(String[] args) {
        // Initial Configuration
        ///String datasetSource = "./data/unc-msd.dat";
        //String datasetLemmPreFilter = "./data/0.4-unc-msd-lemmatized.dat";
        String datasetPath = "./data/0.4-unc-msd.dat";
        boolean lemmatizePages = true;
        String datasetLemmatizedPath = "./data/0.4-unc-msd-lemmatized-banned.dat";
        boolean trainDocRepModel = true;
        String modelDocRepPath = "./models/0.4-docRep-banned";
        boolean trainClusteringModel = false;
        String modelClusteringPath = "./models/0.1-k-means-cluster";

        // Choose test parameters
        int k = 10;
        int documentRepresentation = WORD_2_VEC;
        int clusteringType = CLUSTERING_K_MEANS;
        boolean printClusters = true;
        int distanceType = EUCLEDIAN_DISTANCE;
        int mappingType = MAP_MOST_FREQUENT;
        int kFreqCateg = 3;

        //WORD2VEC parameters
        int vectorSize = 200;
        int numPartitions = 8;
        int numIterations = 8;
        int wordMinCount = 5;

        // Initialize Spark
        SparkConf sparkConf = new SparkConf(true).setAppName("Word count optimization");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);
        sc.setLogLevel("ERROR");


        // Compute document representation
        Preprocessing pre = new Preprocessing(sc, distanceType);
        JavaRDD<TextWikiPage> dDataset;
        DocumentRepresentation docRep = null;
        switch (documentRepresentation) {
            case BAG_OF_WORDS:
                docRep = new BagOfWordsSpark(sc, vectorSize, wordMinCount);
                break;
            case WORD_2_VEC:
                docRep = new DocumentRepresentationWord2Vec(sc, vectorSize, numPartitions, numIterations, wordMinCount);
                break;
        }

        /*JavaRDD<WikiPage> dFullDataset = pre.loadFull(datasetSource);
        System.out.println("FULL:" +dFullDataset.count());
        JavaRDD<TextWikiPage> dLemmatizedWikiPagesOld = pre.loadSplitted(datasetLemmPreFilter);
        JavaRDD<WikiPage> dSample = Preprocessing.extractDocuments(dFullDataset,dLemmatizedWikiPagesOld);
        pre.save(dSample,datasetPath);*/


        //System.out.println("Lemmatize and Filter WikiPages");
        if (lemmatizePages) {
            //System.out.println("Peforming lemmatization...");
            //Load WikiPages and perform lemmatization
            Lemmatizer.setSc(sc);
            Lemmatizer.generateBannedCategorySet();

            JavaRDD<WikiPage> dLemmatizedWikiPages = pre.lemmatize(datasetPath).cache();
            //Save the lemmatized pages
            pre.save(dLemmatizedWikiPages, datasetLemmatizedPath);
            //Split the WikiPages
            dDataset = pre.splitWikiPages(dLemmatizedWikiPages).cache();
        } else {
            //System.out.println("Loading lemmatized dataset...");
            dDataset = pre.loadSplitted(datasetLemmatizedPath).cache();
        }

        //System.out.println("Training DocumentRepresentation model");
        switch (documentRepresentation) {
            case BAG_OF_WORDS:
                //System.out.println("Training model...");
                docRep.trainModel(dDataset.map(TextWikiPage::getText).cache());
                break;
            case WORD_2_VEC:
                if (trainDocRepModel) {
                    //System.out.println("Training model...");
                    docRep.trainModel(dDataset.map(TextWikiPage::getText).cache());
                    docRep.saveModel(modelDocRepPath);
                } else {
                    //System.out.println("Loading model...");
                    docRep.loadModel(modelDocRepPath);
                }
                break;
        }

        System.out.println("#########################################################################");

        docRep.distributeModel();
        JavaRDD<VectorWikiPage> dFullVectorPages = null;
        JavaPairRDD<String, Tuple2<Vector, Integer>> dVectorCategory = null;
        Broadcast<Object2ObjectOpenHashMap<String, Tuple2<Vector, Integer>>> bCategoryMap = null;
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
                if (trainClusteringModel) {
                    //System.out.println("Performing K-Means train...");
                    JavaRDD<Vector> dVectors = dVectorPages.map(VectorWikiPage::getVector);
                    kMeans.trainModel(dVectors.rdd().cache(), k, 20);
                    kMeans.saveModel(modelClusteringPath);
                } else {
                    //System.out.println("Performing K-Means load...");
                    kMeans.loadModel(modelClusteringPath);
                }
                kMeans.distributeModel();
                dVectorsClusters = ClusteringKMeans.predict(dVectorPages);
                break;
            case CLUSTERING_LDA:
                //System.out.println("LDA clustering...");
                ClusteringLDA clda = new ClusteringLDA(sc, dVectorPages);
                //System.out.println("Training LDA model...");
                clda.trainModel(k, 20);
                dVectorsClusters = clda.getClustering();
                break;
        }

        /*
        // Group the elements og the clusters
        System.out.println("\nGrouping the elements of the same cluster...");
        JavaPairRDD<Long, Iterable<VectorWikiPage>> dClusteredDataset = dVectorsClusters
                .groupByKey()
                .sortByKey()
                .cache();


        // Print clusters
        if (printClusters) {
            System.out.println("\nClusters: ");
            for (Tuple2<Long, Iterable<VectorWikiPage>> cluster : dClusteredDataset.collect()) {

                System.out.print("Cluster -> [(" + cluster._1 + "): ");
                for (VectorWikiPage element : cluster._2) {
                    System.out.print(element.getId() + " ");
                }
                System.out.print("\b]\n");
            }
            System.out.println();
        }
        */

        //SUPERVISED TESTS
        SupervisedEvaluation se = new SupervisedEvaluation(sc, distanceType);
        JavaPairRDD<Long, Integer> clustersSize = se.computeClusterSize(dVectorsClusters);
        ArrayList<Tuple2<Long, Integer>> lClusterSize = new ArrayList<>(clustersSize.sortByKey().collect());
        Broadcast<ArrayList<Tuple2<Long, Integer>>> bClusterSize = sc.broadcast(lClusterSize);


        //System.out.println("\nRAND AND JACCARD");
        String[] lRandAndJaccard = se.printRandAndJaccard(dVectorsClusters, bClusterSize);

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

        System.out.println(k + ";" + wordMinCount + ";" + lSeparation[0] + ";" + lCohesion[0] + ";" + lSilhouette[0] + ";" + lRandAndJaccard[0]
                + ";" + lRandAndJaccard[1] + ";" + lEntropy[0] + ";" + lEntropy[1] + ";" + outClusterSize + ";" + lCohesion[1] + ";" + lSilhouette[1] + ";"
                + lEntropy[2]);

    }
}
