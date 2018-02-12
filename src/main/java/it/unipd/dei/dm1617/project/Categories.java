package it.unipd.dei.dm1617.project;

import java.io.Serializable;
import java.util.ArrayList;

public class Categories implements Serializable {
    private ArrayList<String> categories;


    public Categories(ArrayList<String> categories) {
        this.categories = categories;
    }

    public void setCategories(ArrayList<String> categories) {
        this.categories = categories;
    }

    public ArrayList<String> toArrayList() {
        return categories;
    }

    @Override
    public boolean equals(Object obj) {
        Categories out = (Categories) obj;

        if (out.categories.size() != this.categories.size())
            return false;

        for (int i = 0; i < this.categories.size(); ++i)
            if (!out.categories.get(i).equals(this.categories.get(i)))
                return false;

        return true;
    }
}
