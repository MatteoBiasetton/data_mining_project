package it.unipd.dei.dm1617.project;

import it.unimi.dsi.fastutil.objects.Object2ObjectOpenHashMap;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.mllib.linalg.Vector;
import scala.Tuple2;

import java.util.*;

import org.apache.spark.broadcast.Broadcast;


public class SupervisedEvaluation {

    private static JavaSparkContext sc;
    private static int distanceType;

    SupervisedEvaluation(JavaSparkContext in_sc, int in_distanceType) {
        sc = in_sc;
        distanceType = in_distanceType;
    }

    /**
     * Compute the binomial of the sum of the given numbers.
     *
     * @param pi Input numbers.
     * @return Binomial of the sum of the given numbers.
     */
    private static long binomialOfSums(Iterable<Long> pi) {
        long sum = 0;
        for (Long p : pi) {
            sum += p;
        }
        return binomial(sum, 2);
    }

    /**
     * Compute the sum of the binomial of the given numbers.
     *
     * @param pi Input numbers.
     * @return Sum of the binomial of the given numbers.
     */
    private static long sumOfBinomials(Iterable<Long> pi) {
        long sum = 0;
        for (Long p : pi) {
            sum += binomial(p, 2);
        }
        return sum;
    }

    /**
     * Calculate the binomial coefficient of two integers
     *
     * @param n First number.
     * @param k Second number.
     * @return The binomial of the two numbers.
     */
    private static long binomial(long n, long k) {
        if (k > n - k)
            return 0;
        //k = n - k;

        long b = 1;
        for (long i = 1, m = n; i <= k; i++, m--)
            b = b * m / i;
        return b;
    }

    /**
     * Compute the size of each cluster.
     *
     * @param dClusteredDataset [ID_CLUSTER, PAGE]
     * @return [ID_CLUSTER, #elements in the cluster ID_CLUSTER]
     */
    public JavaPairRDD<Long, Integer> computeClusterSize(JavaPairRDD<Long, VectorWikiPage> dClusteredDataset)
    {
        return dClusteredDataset
                .mapToPair((doc) -> (new Tuple2<>(doc._1, 1))) //[ID_CLUSTER,1]
                .reduceByKey((x, y) -> x + y); //Count the elements belonging to each cluster
    }

    /**
     * For each cluster compute the frequency of each category within the cluster.
     *
     * @param dClusteredDataset [ID_CLUSTER,TRANSFORMED_PAGE]
     * @return dClusterCategoryFrequency: [ID_CLUSTER,{Category,#occurrences}]
     */
    public JavaPairRDD<Long, Tuple2<String, Integer>> computeClusterCategoryFrequency(JavaPairRDD<Long, VectorWikiPage> dClusteredDataset)
    {
        return dClusteredDataset
                .flatMapToPair((doc) -> {
                    ArrayList<String> lCat = doc._2.getCategories();
                    ArrayList<Tuple2<Long, String>> lSplitCat = new ArrayList<>(lCat.size());

                    for (String c : lCat)
                        lSplitCat.add(new Tuple2<>(doc._1, c));
                    return lSplitCat.iterator();
                })                                              //[ID_CLUSTER, Category]
                .mapToPair((obj) -> new Tuple2<>(obj, 1))   //[{ID_CLUSTER,Category},1]
                .reduceByKey((x, y) -> x + y)                   //[{ID_CLUSTER,Category}, #occurrences]
                .mapToPair((tupla) -> new Tuple2<>(tupla._1._1, new Tuple2<>(tupla._1._2, tupla._2))); //[ID_CLUSTER,{Category,#occurrences}]
    }

    /**
     * Compute the entropy of a cluster.
     * @param dClusterCategoryFrequency [ID_CLUSTER, {CATEGORY, #occurrences in ID_CLUSTER}]
     * @param bClusterSize Broadcast [ID_CLUSTER, Size of cluster ID_CLUSTER], sorted by ID_CLUSTER
     * @return dClusterEntropy: [ID_CLUSTER, Entropy of cluster ID_CLUSTER]
     */
    public JavaPairRDD<Long, Double> clusterEntropy(JavaPairRDD<Long, Tuple2<String, Integer>> dClusterCategoryFrequency,
                                                    Broadcast<ArrayList<Tuple2<Long, Integer>>> bClusterSize)
    {
        return dClusterCategoryFrequency
                .mapToPair((clusterCategory) ->
                {
                    ArrayList<Tuple2<Long, Integer>> lClusterSize = bClusterSize.getValue();
                    double occurrences = (double) clusterCategory._2._2;
                    int idCluster = (clusterCategory._1).intValue();
                    double clusterSize = lClusterSize.get(idCluster)._2.doubleValue();

                    double entropyCoeff = -((occurrences / clusterSize) * ((Math.log10(occurrences / clusterSize)) / (Math.log10(2))));

                    if (entropyCoeff == -0.0)
                        entropyCoeff = 0.0;

                    return new Tuple2<>(clusterCategory._1, entropyCoeff);
                }) //[ID_CLUSTER, entropy coefficient of a category in ID_CLUSTER]
                .reduceByKey((x, y) -> x + y); //[ID_CLUSTER, entropy of cluster ID_CLUSTER]
    }

    public double avgClusterEntropy(JavaPairRDD<Long, Double> dClusterEntropy, int k) {
        return dClusterEntropy
                .map((cluster) -> cluster._2)
                .reduce((x, y) -> x + y) / k;
    }


    /**
     * Compute the entropy of a category.
     *
     * @param dClusterCategoryFrequency [ID_CLUSTER, {CATEGORY, #occurrences in ID_CLUSTER}]
     * @param bCategoryMap  Broadcast Map{CATEGORY, #occurrences in the entire dataset}
     * @return dCategoryEntropy: [CATEGORY, Entropy of CATEGORY]
     */
    public JavaPairRDD<String, Double> categoryEntropy(JavaPairRDD<Long, Tuple2<String, Integer>> dClusterCategoryFrequency,
                                                       Broadcast<Object2ObjectOpenHashMap<String, Tuple2<Vector, Integer>>> bCategoryMap)
    {
        return dClusterCategoryFrequency
                .mapToPair((clusterCategory) ->
                {
                    Object2ObjectOpenHashMap<String, Tuple2<Vector, Integer>> lCategoryMap = bCategoryMap.getValue();
                    double occurrences = (double) clusterCategory._2._2;
                    double categoryFrequency = (double) lCategoryMap.get(clusterCategory._2._1)._2;
                    double entropyCoeff = -((occurrences / categoryFrequency) * ((Math.log10(occurrences / categoryFrequency)) / (Math.log10(2))));

                    if (entropyCoeff == -0.0)
                        entropyCoeff = 0.0;
                    return new Tuple2<>(clusterCategory._2._1, entropyCoeff);
                }) //[CATEGORY i, entropy coefficient category i] for each cluster
                .reduceByKey((x, y) -> x + y); //[CATEGORY, entropy of category]

    }

    public double avgCategoryEntropy(JavaPairRDD<String, Double> dCategoryEntropy,
                                     Broadcast<Object2ObjectOpenHashMap<String, Tuple2<Vector, Integer>>> bCategoryMap) {
        return dCategoryEntropy
                .map((cluster) -> cluster._2)
                .reduce((x, y) -> x + y) / bCategoryMap.value().size();
    }

    /**
     * Calculate the Rand Statistic and the Jaccard Coefficient of the given clustered pages.
     *
     * @param dClusters The clustered Pages. [ID_CLUSTER, ID_PAGE]
     * @param bClusterSize Broadcast [ID_CLUSTER, Size of cluster ID_CLUSTER]
     * @return [Rand Statistic, Jaccard Coefficient]
     */
    private double[] RandAndJaccardCoefficients(JavaPairRDD<Long, VectorWikiPage> dClusters,
                                                Broadcast<ArrayList<Tuple2<Long, Integer>>> bClusterSize) {

        //[[IdCluster,IdCategory], 1]
        final JavaPairRDD<Tuple2<Long, String>, Long> dClusterCategory = dClusters
                .flatMapToPair((page) ->
                {
                    ArrayList<Tuple2<Long, String>> clusterCategories = new ArrayList<>(page._2.getCategories().size());

                    for (String cat : page._2.getCategories()) {
                        clusterCategories.add(new Tuple2<>(page._1, cat));
                    }

                    return clusterCategories.iterator();
                })
                .mapToPair((clusterCat) -> new Tuple2<>(clusterCat, (long) 1));

        long totalSize = 0;

        for (Tuple2<Long, Integer> clusterSize : bClusterSize.getValue())
            totalSize += clusterSize._2;

        long distinctPairs = binomial(totalSize, 2);

        double distinctPairsSameClass = fA(dClusters);

        double distinctPairsDiffClass = distinctPairs - distinctPairsSameClass;

        double f01 = f01(dClusterCategory);

        double f11 = f11(dClusterCategory);

        double f10 = distinctPairsSameClass - f11;

        double f00 = distinctPairsDiffClass - f01;

        double rand = (f00 + f11) / (f00 + f11 + f01 + f10);

        double jaccard = (f11) / (f11 + f01 + f10);

        return new double[]{rand, jaccard};
    }

    /**
     * Number of pairs of points of the same class in the same cluster.
     *
     * @param dClusterCategory [[IdCluster,IdCategory], 1]
     * @return F11 Number of pairs of points of the same class in the same cluster.
     */
    private long f11(JavaPairRDD<Tuple2<Long, String>, Long> dClusterCategory) {
        return dClusterCategory
                .reduceByKey((x, y) -> x + y)
                .mapToPair((tupla) -> new Tuple2<>(tupla._1, binomial(tupla._2, 2)))
                .values()
                .filter((x) -> x > 0)
                .reduce((a, b) -> a + b);
    }

    /**
     * Number of pairs of points of distinct classes in the same cluster.
     *
     * @param dClusterCategory [[IdCluster,IdCategory], 1]
     * @return F01 Number of pairs of points of distinct classes in the same cluster.
     */
    private long f01(JavaPairRDD<Tuple2<Long, String>, Long> dClusterCategory) {

        return dClusterCategory
                .reduceByKey((x, y) -> x + y)                               //<IdCluster.IdCategory, occurrenceCount>
                .mapToPair((tupla) -> new Tuple2<>(tupla._1._1, tupla._2))   //<IdCluster, occurrenceCount>
                .groupByKey()                                               //<IdCluster, Iterable<occurrenceCount>>
                .mapToPair((tupla) -> new Tuple2<>(tupla._1, binomialOfSums(tupla._2) - sumOfBinomials(tupla._2)))  //<IdCluster, pairsInACluster>>
                .values()
                .reduce((a, b) -> a + b);
    }

    /**
     * Compute the number of distinct pairs of the same class in the entire dataset.
     *
     * @param dClusters
     * @return
     */
    private long fA(JavaPairRDD<Long, VectorWikiPage> dClusters) {
        return dClusters
                .mapToPair((page) -> new Tuple2<>(page._2.getCategories().get(0), 1))
                .reduceByKey((x, y) -> x + y)
                .map((cat) -> binomial(cat._2, 2))
                .reduce((x, y) -> x + y);

    }

    /**
     * @param dVectorsClusters
     * @param bClusterSize
     * @return [Rand, Jaccard]
     */
    String[] printRandAndJaccard(JavaPairRDD<Long, VectorWikiPage> dVectorsClusters,
                                 Broadcast<ArrayList<Tuple2<Long, Integer>>> bClusterSize) {
        //System.out.println("Computing Jaccard coefficient...");
        double[] stat = RandAndJaccardCoefficients(dVectorsClusters, bClusterSize);

        String[] out = new String[2];
        out[0] = stat[0] + "";
        out[1] = stat[1] + "";

        //System.out.println("Rand coefficient: " + stat[0]);
        //System.out.println("Jaccard coefficient: " + stat[1]);

        return out;
    }

    /**
     *
     * @param dVectorsClusters
     * @param bCategoryMap
     * @param bClusterSize
     * @param k
     * @return [AvgClusterEntropy, AvgCategoryEntropy, Cluster-Entropy]
     */
    String[] printEntropy(JavaPairRDD<Long, VectorWikiPage> dVectorsClusters,
                          Broadcast<Object2ObjectOpenHashMap<String, Tuple2<Vector, Integer>>> bCategoryMap,
                      Broadcast<ArrayList<Tuple2<Long, Integer>>> bClusterSize,
                      int k) {
        JavaPairRDD<Long, Tuple2<String, Integer>> dClusterCategoryFrequency = computeClusterCategoryFrequency(dVectorsClusters).cache();

        JavaPairRDD<String, Double> dCategoryEntropy = categoryEntropy(dClusterCategoryFrequency, bCategoryMap);
        JavaPairRDD<Long, Double> dClusterEntropy = clusterEntropy(dClusterCategoryFrequency, bClusterSize);


        //System.out.println("Computing category entropy...");
        //final List<Tuple2<String, Double>> lCategoryEntropy = dCategoryEntropy.collect();

        //System.out.println("Computing cluster entropy...");
        final List<Tuple2<Long, Double>> lClusterEntropy = dClusterEntropy.sortByKey().collect();


        /*
        //System.out.println("Category Entropy");
        System.out.println("Category - Entropy");
        for (Tuple2<String, Double> e : lCategoryEntropy)
            System.out.println(e._1 + " - " + e._2);*/


        String[] out = new String[3];

        out[0] = "" + avgClusterEntropy(dClusterEntropy, k);
        out[1] = "" + avgCategoryEntropy(dCategoryEntropy, bCategoryMap);

        StringBuilder output = new StringBuilder();
        for (Tuple2<Long, Double> e : lClusterEntropy)
            output.append(e._2).append(";");

        out[2] = output.substring(0, output.length() - 1);

        return out;
    }
}
