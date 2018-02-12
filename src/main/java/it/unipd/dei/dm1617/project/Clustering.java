package it.unipd.dei.dm1617.project;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;

/**
 * Public interface for clustering
 */
public interface Clustering {
    /**
     * Returns the list of documents with the corresponding cluster index
     *
     * @param dataset [DOCUMENT]
     * @param k       number of cluster
     * @return [ID_CLUSTER, DOCUMENT]
     */
    JavaPairRDD<Long, VectorWikiPage> getClusters(JavaRDD<VectorWikiPage> dataset, int k);
}
