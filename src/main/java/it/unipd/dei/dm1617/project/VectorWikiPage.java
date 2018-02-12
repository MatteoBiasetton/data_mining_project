package it.unipd.dei.dm1617.project;

import org.apache.spark.sql.Encoder;
import org.apache.spark.sql.Encoders;
import org.apache.spark.mllib.linalg.Vector;

import java.io.Serializable;
import java.util.ArrayList;

public class VectorWikiPage implements Serializable
{

    private long id;
    private ArrayList<String> categories;
    private Vector vector;

    public VectorWikiPage()
    {
    }

    public VectorWikiPage(long id, ArrayList<String> categories, Vector vector)
    {
        this.id = id;
        this.categories = categories;
        this.vector = vector;
    }

    public static Encoder<VectorWikiPage> getEncoder() {
        return Encoders.bean(VectorWikiPage.class);
    }

    public long getId()
    {
        return id;
    }

    public void setId(long id)
    {
        this.id = id;
    }

    public ArrayList<String> getCategories()
    {
        return categories;
    }

    public void setCategories(ArrayList<String> categories)
    {
        this.categories = categories;
    }

    public Vector getVector()
    {
        return vector;
    }

    public void setVector(Vector vector)
    {
        this.vector = vector;
    }

    @Override
    public String toString()
    {
        return "ID: " + id + " CATEGORIES: " + categories + " VECTOR: " + vector;
    }

    @Override
    public boolean equals(Object obj)
    {
        return this.id == ((VectorWikiPage) obj).id;
    }
}
