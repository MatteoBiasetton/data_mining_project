package it.unipd.dei.dm1617.project;

import it.unipd.dei.dm1617.Distance;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import scala.Tuple2;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static it.unipd.dei.dm1617.project.SerializableComparator.serialize;


/**
 * Class that implements the k-centers clustering techniques
 */
public class ClusteringKCenters implements Clustering {

    private static int distanceType;
    private static JavaSparkContext sc;

    /**
     * Public constructor for class {@link ClusteringKCenters}
     *
     * @param in_sc {@link JavaSparkContext} used to distribute the computations
     */
    ClusteringKCenters(JavaSparkContext in_sc, int in_distanceType) {
        sc = in_sc;
        distanceType = in_distanceType;
    }

    @Override
    public JavaPairRDD<Long, VectorWikiPage> getClusters(JavaRDD<VectorWikiPage> P, int k) {
        JavaRDD<VectorWikiPage> dRandomSample = P.sample(false, 0.01, System.currentTimeMillis());
        List<VectorWikiPage> lRandomsample = dRandomSample.collect();
        Random rand = new Random(System.currentTimeMillis());
        final VectorWikiPage center = lRandomsample.get(rand.nextInt(lRandomsample.size()));
        ArrayList<Tuple2<Long, VectorWikiPage>> S = new ArrayList<>();
        S.add(new Tuple2<>(0L, center));

        // Compute all the distance from the first center
        JavaRDD<KCenterPoint> P_S = P.map((page) -> {
            double distance = Distance.distance(page.getVector(), center.getVector(), distanceType);
            return new KCenterPoint(page, 0L, distance, distance);
        }).filter((point) -> point.page.getId() != center.getId()).cache();

        for (long i = 1; i < k; i++) {
            // Find element at max distance
            //System.out.println("Center " + i);

            final long current_k = i;

            VectorWikiPage newCenter = P_S.max(serialize(KCenterPoint::compareTo)).page;
            S.add(new Tuple2<>(current_k, newCenter));


            // Update the distances
            P_S = P_S.map((point) -> {
                double distance = Distance.distance(point.page.getVector(), newCenter.getVector(), distanceType);
                if (distance < point.distanceFromCenter) {
                    return new KCenterPoint(point.page, current_k, distance, distance + point.totalDistance);
                } else {
                    return new KCenterPoint(point.page, point.idCenter, point.distanceFromCenter, distance + point.totalDistance);
                }
            }).filter((point) -> point.page.getId() != newCenter.getId()).cache();
        }

        JavaPairRDD<Long, VectorWikiPage> dVectorsCenters = sc.parallelizePairs(S);
        return P_S.mapToPair((point) -> new Tuple2<>(point.idCenter, point.page)).union(dVectorsCenters);
    }


}
