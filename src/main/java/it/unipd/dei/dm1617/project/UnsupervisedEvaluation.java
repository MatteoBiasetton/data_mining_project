package it.unipd.dei.dm1617.project;

import it.unipd.dei.dm1617.Distance;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.Vectors;
import scala.Tuple2;
import scala.Tuple3;

import java.util.*;

public class UnsupervisedEvaluation {

    private static JavaSparkContext sc;
    private static int distanceType;
    private static int k;

    UnsupervisedEvaluation(JavaSparkContext in_sc, int in_distanceType, int in_k)
    {

        sc = in_sc;
        distanceType = in_distanceType;
        k = in_k;
    }

    int getDistanceType() {
        return distanceType;
    }

    void setDistanceType(int distanceType) {
        this.distanceType = distanceType;
    }

    int getK() {
        return k;
    }

    void setK(int in_k) {
        k = in_k;
    }

    /**
     * Compute the intra cluster averages.
     *
     * @param dClusteredDataset [ID_CLUSTER, PAGE]
     * @param bClusterSize      Broadcast [ID_CLUSTER, Size of cluster ID_CLUSTER], sorted by ID_CLUSTER
     * @return dExtraClusterAverages: [ID_PAGE, Intra cluster average]
     */
    public JavaPairRDD<Long, Double> computeIntraClusterAverages(JavaPairRDD<Long, VectorWikiPage> dClusteredDataset,
                                                                 Broadcast<ArrayList<Tuple2<Long, Integer>>> bClusterSize) {

        /* STEP A - For the ith object, calculate its average distance to all other objects in
        *  its cluster. Call this value ai.
        */

        JavaPairRDD<Long, Tuple2<VectorWikiPage, VectorWikiPage>> dVectorPairs = dClusteredDataset.join(dClusteredDataset);

        return dVectorPairs
                .mapToPair((pairPages) ->
                {
                    Vector v1 = pairPages._2._1.getVector();
                    Vector v2 = pairPages._2._2.getVector();
                    return new Tuple2<>(new Tuple2<>(pairPages._1, pairPages._2._1.getId()), Distance.distance(v1, v2, distanceType));
                }) //[{ID_CLUSTER, ID_PAGE}, Distance]
                .reduceByKey((x, y) -> x + y) //[{ID_CLUSTER, ID_PAGE}, Sum of distances]
                .mapToPair((sum) -> {
                    int clusterSize = bClusterSize.getValue().get(sum._1._1.intValue())._2;

                    return new Tuple2<>(sum._1._2, sum._2 / (clusterSize - 1));
                });//[ID_PAGE, Average distance]
    }

    /**
     * Compute the extra cluster averages.
     *
     * @param dClusteredDataset [ID_CLUSTER, PAGE]
     * @param bClusterSize      Broadcast [ID_CLUSTER, Size of cluster ID_CLUSTER], sorted by ID_CLUSTER
     * @return dExtraClusterAverages: [ID_PAGE, Extra cluster average]
     */
    public JavaPairRDD<Long, Double> computeExtraClusterAverages(JavaPairRDD<Long, VectorWikiPage> dClusteredDataset,
                                                                 Broadcast<ArrayList<Tuple2<Long, Integer>>> bClusterSize) {
        /* STEP B - For the ith object and any cluster not containing the object, calculate
        *  the object’s average distance to all the objects in the given cluster. Find
        *  the minimum such value with respect to all clusters; call this value bi.
        */

        return dClusteredDataset
                .cartesian(dClusteredDataset) //Compute the cartesian product: JavaPairRDD([ID_CLUSTER, PAGE], [ID_CLUSTER,PAGE])
                .filter((pair) -> {
                    long idCluster1 = pair._1._1;
                    long idCluster2 = pair._2._1;

                    return idCluster1 != idCluster2;
                }) //Remove the pairs belonging to the same cluster
                .mapToPair((pair) -> {
                    long idCluster1 = pair._1._1;
                    VectorWikiPage page1 = pair._1._2;

                    long idCluster2 = pair._2._1;
                    VectorWikiPage page2 = pair._2._2;

                    return new Tuple2<>(new Tuple3<>(idCluster1, page1.getId(), idCluster2), Distance.distance(page1.getVector(), page2.getVector(), distanceType));
                }) //[{ID_CLUSTER1, ID_PAGE, ID_CLUSTER2},Distance]
                .reduceByKey((x, y) -> x + y) ////[{ID_CLUSTER1, ID_PAGE, ID_CLUSTER2}, Sum of distances to ID_PAGE wrt pages of ID_CLUSTER2]
                .mapToPair((Tuple3) -> {
                    int clusterSize = bClusterSize.getValue().get(Tuple3._1._3().intValue())._2;

                    return new Tuple2<>(Tuple3._1._2(), Tuple3._2 / clusterSize);
                }) //[ID_PAGE, AvgDistance]
                .reduceByKey(Math::min);//[ID_PAGE, MinAvgDistance]
    }

    /**
     * Compute the silhouette coefficient for each page.
     *
     * @param dIntraClusterAverages [ID_PAGE, Intra cluster average]
     * @param dExtraClusterAverages [ID_PAGE, Extra cluster average]
     * @return dSilhouetteCoefficient: [ID_PAGE, Silhouette coefficient]
     */
    public JavaPairRDD<Long, Double> computeSilhouetteCoefficient(JavaPairRDD<Long, Double> dIntraClusterAverages,
                                                                  JavaPairRDD<Long, Double> dExtraClusterAverages) {
        // STEP C - For the ith object, the silhouette coefficient is si = (bi − ai)/ max(ai,bi).

        return dIntraClusterAverages
                .join(dExtraClusterAverages) //[ID_PAGE, {Intra cluster avg, Extra cluster avg}]
                .mapToPair((page) ->
                {
                    double intra = page._2._1;
                    double extra = page._2._2;

                    double coefficient = (extra - intra) / (Math.max(extra, intra));

                    return new Tuple2<>(page._1, coefficient);
                });
    }

    /**
     * Compute the average silhouette coefficient for each cluster.
     *
     * @param bClusterSize           Broadcast [ID_CLUSTER, Size of cluster ID_CLUSTER], sorted by ID_CLUSTER
     * @param dClusteredDataset      [ID_CLUSTER, PAGE]
     * @param dSilhouetteCoefficient [ID_CLUSTER, Silhouette coefficient]
     * @return dAvgSilhouetteCoefficient: [ID_CLUSTER, Avg silhouette]
     */
    public JavaPairRDD<Long, Double> averageSilhouetteCoefficient(Broadcast<ArrayList<Tuple2<Long, Integer>>> bClusterSize,
                                                                  JavaPairRDD<Long, VectorWikiPage> dClusteredDataset,
                                                                  JavaPairRDD<Long, Double> dSilhouetteCoefficient) {

        return dClusteredDataset
                .mapToPair((cluster) -> new Tuple2<>(cluster._2.getId(), cluster._1))//[ID_PAGE, ID_CLUSTER]
                .join(dSilhouetteCoefficient) //[ID_PAGE, {ID_CLUSTER,Silhouette}]
                .mapToPair((tuple) -> tuple._2)//[ID_CLUSTER,Silhouette]
                .reduceByKey((x, y) -> x + y) //[ID_CLUSTER, Sum of silhouette coefficients]
                .mapToPair((cluster) -> {
                    int clusterSize = bClusterSize.getValue().get(cluster._1.intValue())._2;
                    return new Tuple2<>(cluster._1, cluster._2 / clusterSize);
                }); //[ID_CLUSTER, Average silhouette]
    }

    /**
     * Compute the average cluster silhouette.
     *
     * @param dAvgSilhouetteCoefficient [ID_CLUSTER, Avg cluster silhouette]
     * @param bClusterSize              Broadcast [ID_CLUSTER, Size of cluster ID_CLUSTER], sorted by ID_CLUSTER
     * @return Average silhouette.
     */
    public double datasetAverageSilhouette(JavaPairRDD<Long, Double> dAvgSilhouetteCoefficient,
                                           Broadcast<ArrayList<Tuple2<Long, Integer>>> bClusterSize) {


        JavaPairRDD<Long, Double> filter = dAvgSilhouetteCoefficient
                .mapToPair((value) -> {
                    int clusterSize = bClusterSize.getValue().get(value._1.intValue())._2;
                    return new Tuple2<>(value._1, value._2 * clusterSize);
                })
                .filter((coeff) -> !Double.isNaN(coeff._2) && Double.isFinite(coeff._2));

        int totalRemainingPoints = filter
                .map((value) -> bClusterSize.getValue().get(value._1.intValue())._2)
                .reduce((x, y) -> x + y);

        return filter
                .map((value) -> value._2)
                .reduce((x, y) -> x + y) / totalRemainingPoints;
    }

    /**
     * Generate a RDD of random Vectors, deeply stored inside the class VectorWikiPage
     *
     * @param points The Dataset.
     * @param maxVal Max value of the component.
     * @return Set of random vectors.
     */
    private JavaRDD<VectorWikiPage> generateRandVectors(JavaRDD<VectorWikiPage> points, double maxVal) {
        return points.map((point) -> {  //[map a point to a random point]
            double[] components = new double[point.getVector().size()];
            Random generator = new Random(System.currentTimeMillis());
            for (int i = 0; i < point.getVector().size(); i++) {
                components[i] = (generator.nextDouble() * 2 * maxVal) - maxVal;
            }
            VectorWikiPage randVectorPage = new VectorWikiPage();
            randVectorPage.setVector(Vectors.dense(components));
            return randVectorPage;
        });
    }


    public JavaPairRDD<Long, Double> clusterCohesion(Broadcast<ArrayList<Tuple2<Long, Integer>>> bClusterSize,
                                                     JavaPairRDD<Long, VectorWikiPage> dClusteredDataset) {
        JavaPairRDD<Long, Tuple2<VectorWikiPage, VectorWikiPage>> dVectorPairs = dClusteredDataset.join(dClusteredDataset);

        return dVectorPairs
                .mapToPair((pair) -> {
                    VectorWikiPage page1 = pair._2._1;
                    VectorWikiPage page2 = pair._2._2;

                    return new Tuple2<>(pair._1, Distance.distance(page1.getVector(), page2.getVector(), distanceType));
                }) //[ID_CLUSTER, Pair distance]
                .reduceByKey((x, y) -> (x + y))
                .mapToPair((sum) -> {
                    int clusterSize = bClusterSize.getValue().get(sum._1.intValue())._2;

                    return new Tuple2<>(sum._1, sum._2 / (clusterSize * (clusterSize - 1)));
                });
    }

    /**
     * Average over all cluster cohesion coefficients.
     *
     * @param dClusterCohesion [ID_CLUSTER, Cohesion of cluster ID_CLUSTER]
         * @return Average cohesion.
         */
    public double datasetCohesion(JavaPairRDD<Long, Double> dClusterCohesion) {

        JavaRDD<Double> dFiltered = dClusterCohesion
                .map((cluster) -> cluster._2)
                .filter((val) -> !Double.isNaN(val) && Double.isFinite(val));

        long count = dFiltered.count();

        return dFiltered
                .reduce((x, y) -> x + y) / count;


    }

    /**
     * From the given dataset of VectorWikiPage return the max value from all the components of the vectors.
     * This number is later used to generate the random vectors.
     *
     * @param points The Dataset.
     * @return The max value of the component.
     */
    private double maxComponentValue(JavaRDD<VectorWikiPage> points) {
        return points.map((point) -> { //[RDD of max components]
            double max = 0;
            for (int i = 0; i < point.getVector().size(); i++)
                if (point.getVector().apply(i) > max)
                    max = point.getVector().apply(i);
            return max;
        }).reduce(Math::max);   //[return the max]
    }

    /**
     * Calculate the Hopkin Statistic of a given dataset.
     *
     * @param dVectorPages The Dataset.
     * @return The Hopkin Statistic.
     */
    public double hopkinStatistic(JavaRDD<VectorWikiPage> dVectorPages) {
        JavaRDD<VectorWikiPage> dVectorsT = dVectorPages.sample(false, 0.1);
        JavaRDD<VectorWikiPage> dRandVectorsT = generateRandVectors(dVectorsT, maxComponentValue(dVectorsT));
        //JavaRDD<VectorWikiPage> dReducedDataset = dVectorPages.subtract(dVectorsT)
        return hopkin(dVectorPages, dVectorsT, dRandVectorsT);
    }

    /**
     * Calculate the Hopkin Statistic of a given dataset with respect to a random dataset
     *
     * @param dataset The whole Dataset.
     * @param datasetSample The Subset of the Dataset.
     * @param randPoints The Set of random vectors, with the same cardinality of the Subset.
     * @return The Hopkin Statistic.
     */
    private double hopkin(JavaRDD<VectorWikiPage> dataset, JavaRDD<VectorWikiPage> datasetSample, JavaRDD<VectorWikiPage> randPoints) {
        double sumWi = rddDistance(datasetSample, dataset);
        double sumYi = rddDistance(randPoints, dataset);
        return (sumWi / (sumWi + sumYi));
    }

    double rddDistance(JavaRDD<VectorWikiPage> points, JavaRDD<VectorWikiPage> dataset) {
        return points.cartesian(dataset)
                .mapToPair((couple) -> new Tuple2<>(couple._1.getId(), Distance.distance(couple._1.getVector(), couple._2.getVector(), distanceType)))
                .filter((tupla) -> tupla._2 > 0)
                .reduceByKey(Math::min)
                .values()
                .reduce((x, y) -> x + y);
    }


    /**
     * Compute the separation between each cluster pair (ordered).
     *
     * @param bClustersSizes    [ID_CLUSTER, # documents in ID_CLUSTER], sorted by ID_CLUSTER.
     * @param dClusteredDataset [ID_CLUSTER, DOCUMENT]
     * @return dClusterSeparation: [ID_CLUSTER1, ID_CLUSTER2, Separation between ID_CLUSTER1 and ID_CLUSTER2]
     */
    public JavaRDD<Tuple3<Long, Long, Double>> clusterSeparation(JavaPairRDD<Long, VectorWikiPage> dClusteredDataset,
                                                                 Broadcast<ArrayList<Tuple2<Long, Integer>>> bClustersSizes) {
        return dClusteredDataset
                .cartesian(dClusteredDataset) //Compute the cartesian product: JavaPairRDD([ID_CLUSTER, PAGE], [ID_CLUSTER,PAGE])
                .filter((pair) -> {
                    long idCluster1 = pair._1._1;
                    long idCluster2 = pair._2._1;

                    return idCluster1 < idCluster2;
                }) //Remove all the pairs that would be considered twice
                .mapToPair((pair) -> {
                    long idCluster1 = pair._1._1;
                    VectorWikiPage page1 = pair._1._2;

                    long idCluster2 = pair._2._1;
                    VectorWikiPage page2 = pair._2._2;

                    return new Tuple2<>(new Tuple2<>(idCluster1, idCluster2), Distance.distance(page1.getVector(), page2.getVector(), distanceType));
                }) //[{ID_CLUSTER1,ID_CLUSTER2},Distance]
                .reduceByKey((x, y) -> x + y)  //[{ID_CLUSTER1,ID_CLUSTER2},Sum of distances]
                .map((pair) -> {
                    int sizeCluster1 = bClustersSizes.getValue().get(pair._1._1.intValue())._2;
                    int sizeCluster2 = bClustersSizes.getValue().get(pair._1._2.intValue())._2;

                    return new Tuple3<>(pair._1._1, pair._1._2, pair._2 / (sizeCluster1 * sizeCluster2));
                }); //[ID_CLUSTER1,ID_CLUSTER2,Average distance]
    }

    public double datasetSeparation(JavaRDD<Tuple3<Long, Long, Double>> dClusterSeparation, int k) {
        return dClusterSeparation
                .map(Tuple3::_3)
                .reduce((x, y) -> x + y) / (k * (k - 1) / 2);
    }

    public String[] printCohesion(JavaPairRDD<Long, VectorWikiPage> dClusteredDataset,
                                  Broadcast<ArrayList<Tuple2<Long, Integer>>> bClusterSize) {
        JavaPairRDD<Long, Double> dClusterCohesion = clusterCohesion(bClusterSize, dClusteredDataset);

        final List<Tuple2<Long, Double>> lClusterCohesion = dClusterCohesion.sortByKey().collect();

        String[] out = new String[2];
        StringBuilder output = new StringBuilder();
        //System.out.println("Cluster-Cohesion:");
        for (Tuple2<Long, Double> e : lClusterCohesion) {
            output.append(e._2).append(";");
        }

        out[1] = output.substring(0, output.length() - 1);

        //System.out.println("Dataset cohesion: " + datasetCohesion(dClusterCohesion));
        out[0] = "" + datasetCohesion(dClusterCohesion);
        return out;
    }


    public String[] printSilhouette(JavaPairRDD<Long, VectorWikiPage> dClusteredDataset,
                                    Broadcast<ArrayList<Tuple2<Long, Integer>>> bClusterSize) {

        //System.out.println("Computing intra cluster averages...");
        JavaPairRDD<Long, Double> dIntraClusterAverages =
                computeIntraClusterAverages(dClusteredDataset, bClusterSize);

        //System.out.println("Computing extra cluster averages and silhouette coefficients...");

        JavaPairRDD<Long, Double> dExtraClusterAverages =
                computeExtraClusterAverages(dClusteredDataset, bClusterSize);

        JavaPairRDD<Long, Double> dSilhouetteCoefficient = computeSilhouetteCoefficient(dIntraClusterAverages, dExtraClusterAverages);

        JavaPairRDD<Long, Double> dAverageSilhouetteCoefficient = averageSilhouetteCoefficient(bClusterSize, dClusteredDataset, dSilhouetteCoefficient);

        final List<Tuple2<Long, Double>> lAverageSilhouetteCoefficient = dAverageSilhouetteCoefficient.sortByKey().collect();

        //System.out.println("Average silhouette coefficients: ");

        String[] out = new String[2];
        StringBuilder output = new StringBuilder();
        //System.out.println("Cluster-Average silhouette");
        for (Tuple2<Long, Double> e : lAverageSilhouetteCoefficient) {
            output.append(e._2).append(";");
        }

        out[1] = output.substring(0, output.length() - 1);

        //System.out.println("Average silhouette: " + datasetAverageSilhouette(dAverageSilhouetteCoefficient, bClusterSize, k));
        out[0] = "" + datasetAverageSilhouette(dAverageSilhouetteCoefficient, bClusterSize);
        return out;
    }

    public String[] printSeparation(JavaPairRDD<Long, VectorWikiPage> dClusteredDataset,
                                    Broadcast<ArrayList<Tuple2<Long, Integer>>> bClusterSize,
                                    int k) {
        //System.out.println("Computing clusterSeparation...");
        JavaRDD<Tuple3<Long, Long, Double>> dClusterSeparation = clusterSeparation(dClusteredDataset, bClusterSize);

        //final List<Tuple3<Long, Long, Double>> lSeparation = dClusterSeparation.collect();

        String[] out = new String[2];
        //String output = "";
        //System.out.println("Cluster 1 - Cluster 2 - Separation");
        //for (Tuple3<Long, Long, Double> e : lSeparation)
        //output += e._1() + ";" + e._2() + ";" + e._3() + "\n";
        //System.out.println(e._1() + " - " + e._2() + " - " + e._3());
        //out[0] = output;


        //System.out.println("Average cluster separation: " + datasetSeparation(dClusterSeparation, k));
        out[0] = "" + datasetSeparation(dClusterSeparation, k);
        return out;
    }


    public String[] printHopkins(JavaRDD<VectorWikiPage> dVectorPages) {
        //System.out.println("Computing Hopkin's coefficient...");
        //System.out.println("Hopkin's coefficient: " + hopkinStatistic(dVectorPages));
        String[] out = new String[1];
        out[0] = hopkinStatistic(dVectorPages) + "";
        return out;
    }
}
