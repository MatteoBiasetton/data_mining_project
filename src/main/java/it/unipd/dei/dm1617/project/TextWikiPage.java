package it.unipd.dei.dm1617.project;

import org.apache.spark.sql.Encoder;
import org.apache.spark.sql.Encoders;
import java.io.Serializable;
import java.util.ArrayList;

public class TextWikiPage implements Serializable
{

    private long id;
    private ArrayList<String> categories;
    private ArrayList<String> text;

    public TextWikiPage()
    {
    }

    public TextWikiPage(long id, ArrayList<String> categories, ArrayList<String> text)
    {
        this.id = id;
        this.categories = categories;
        this.text = text;
    }

    public static Encoder<TextWikiPage> getEncoder()
    {
        return Encoders.bean(TextWikiPage.class);
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

    public ArrayList<String> getText()
    {
        return text;
    }

    public void setText(ArrayList<String> text)
    {
        this.text = text;
    }

    @Override
    public String toString()
    {
        return "(" + id + ") `" + "` " + categories;
    }

}
