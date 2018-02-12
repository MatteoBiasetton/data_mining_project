package it.unipd.dei.dm1617.project;

import it.unipd.dei.dm1617.Distance;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;
import scala.Tuple2;
import java.util.ArrayList;
import java.util.Random;

/**
 * Class that implements the k-medians clustering techniques
 */
public class ClusteringKMedians implements Clustering {

    private static int distanceType;
    private JavaSparkContext sc;

    /**
     * Public constructor for class {@link ClusteringKMedians}
     *
     * @param sc {@link JavaSparkContext} used to distribute the computations
     */
    ClusteringKMedians(JavaSparkContext sc, int distanceType) {
        this.sc = sc;
        this.distanceType = distanceType;
    }

    @Override
    public JavaPairRDD<Long, VectorWikiPage> getClusters(JavaRDD<VectorWikiPage> dataset, int k) {
        JavaRDD<VectorWikiPage> P = dataset;
        ArrayList<VectorWikiPage> S = new ArrayList<>();
        ArrayList<VectorWikiPage> lDataset = new ArrayList<>(dataset.collect());
        int index;
        Random r = new Random(System.currentTimeMillis());
        // choose the first k points randomly
        for (int i = 0; i < k; i++) {
            index = r.nextInt(lDataset.size());
            S.add(lDataset.get(index));
            lDataset.remove(index);
        }
        ArrayList<VectorWikiPage> P_S;
        ArrayList<VectorWikiPage> S1;
        JavaPairRDD<VectorWikiPage, VectorWikiPage> C = partitions(P, S);
        JavaPairRDD<VectorWikiPage, VectorWikiPage> C1;
        boolean stoppingCondition = false;
        long iteration = 1;
        // While no better clusters are found
        while (!stoppingCondition) {
            stoppingCondition = true;
            P_S = computeP_S(P, S);
            double c_score = objectiveFunction(C);
            // For each point and for each center swap a point with a center and recompute cluster
            for (VectorWikiPage p : P_S) {
                double c1_score = c_score;
                for (VectorWikiPage c : S) {
                    S1 = new ArrayList<>();
                    S1.addAll(S);
                    S1.remove(c);
                    S1.add(p);
                    C1 = partitions(P, S1);
                    c1_score = objectiveFunction(C1);
                    // If a better cluster is found keep it
                    if (c1_score < c_score) {
                        stoppingCondition = false;
                        System.out.println("Found better clustering with objectiveFunction: " + c1_score);
                        C = C1;
                        S = S1;
                        c_score = c1_score;
                        break;
                    }
                    System.out.println("Iteration number: " + iteration++);
                }
                if (c1_score < c_score) {
                    break;
                }
            }
        }
        return C.mapToPair((page) -> new Tuple2<>(page._1.getId(), page._2));
    }

    /**
     * Calculate the objective function for k-medians clustering
     *
     * @param c1 [CENTER_CLUSTER_DOCUMENT, DOCUMENT] list of documents with the corresponding cluster
     * @return objective function computed on the given clusters
     */
    private double objectiveFunction(JavaPairRDD<VectorWikiPage, VectorWikiPage> c1) {
        JavaPairRDD<VectorWikiPage, VectorWikiPage> d_c1 = c1;
        double f;
        JavaRDD<Double> d_cluster = d_c1.map((center2element) ->
                Distance.distance(center2element._1.getVector(), center2element._2.getVector(), distanceType));
        f = d_cluster.reduce((v1, v2) -> v1 + v2);
        return f;
    }

    /**
     * Extract the list of documents that are not centers
     *
     * @param P list of all the documents
     * @param S list of centers
     * @return list of documents that are not centers
     */
    private ArrayList<VectorWikiPage> computeP_S(JavaRDD<VectorWikiPage> P, ArrayList<VectorWikiPage> S) {
        ArrayList<VectorWikiPage> P_S = new ArrayList<>();
        P_S.addAll(P.collect());
        P_S.removeAll(S);
        return P_S;
    }

    /**
     * Compute the clusters given the centers
     * @param P list of all the documents
     * @param S list of centers
     * @return [CLUSTER_ID, DOCUMENT]
     */
    private JavaPairRDD<VectorWikiPage, VectorWikiPage> partitions(JavaRDD<VectorWikiPage> P, ArrayList<VectorWikiPage> S) {
        Broadcast<ArrayList<VectorWikiPage>> S_broadcast = sc.broadcast(S);
        JavaPairRDD<VectorWikiPage, VectorWikiPage> center2elem = P.mapToPair((element) ->
        {
            double distance = 0;
            double minDistance;
            VectorWikiPage candidateCenter = null;
            minDistance = Double.MAX_VALUE;
            for (VectorWikiPage center : S_broadcast.value()) {
                distance = Distance.distance(element.getVector(), center.getVector(), distanceType);
                if (distance < minDistance) {
                    minDistance = distance;
                    candidateCenter = center;
                }
            }
            return new Tuple2<>(candidateCenter, element);
        });
        return center2elem;
    }
}