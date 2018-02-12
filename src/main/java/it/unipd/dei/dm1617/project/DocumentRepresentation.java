package it.unipd.dei.dm1617.project;

import org.apache.spark.api.java.JavaRDD;
import java.io.Serializable;
import java.util.ArrayList;

/**
 * Public interface to abstract a document representation
 */
public interface DocumentRepresentation extends Serializable {

    /**
     * Save the model to the disk
     *
     * @param filename path in which save data
     */
    void saveModel(String filename);

    /**
     * Load a saved model
     *
     * @param filename path of the data
     */
    void loadModel(String filename);

    /**
     * Train the model on a given dataset
     *
     * @param dDataset dataset to use for training
     */
    void trainModel(JavaRDD<ArrayList<String>> dDataset);

    /**
     * Create a distributed version of the model to improve performance
     */
    void distributeModel();
}