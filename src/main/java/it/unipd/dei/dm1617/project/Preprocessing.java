package it.unipd.dei.dm1617.project;

import it.unimi.dsi.fastutil.objects.Object2IntOpenHashMap;
import it.unimi.dsi.fastutil.objects.Object2ObjectOpenHashMap;
import org.apache.spark.api.java.*;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.mllib.linalg.Vector;
import scala.Tuple2;
import it.unipd.dei.dm1617.Distance;
import it.unipd.dei.dm1617.InputOutput;
import it.unipd.dei.dm1617.Lemmatizer;
import it.unipd.dei.dm1617.WikiPage;

import java.util.*;

public class Preprocessing {
    private static JavaSparkContext sc;
    private static int distanceType;

    Preprocessing(JavaSparkContext context, int in_distanceType) {
        sc = context;
        distanceType = in_distanceType;
    }


    public static JavaRDD<WikiPage> extractDocuments(JavaRDD<WikiPage> dFullDataset,
                                                     JavaRDD<TextWikiPage> dLemmatizedWikiPages) {
        JavaPairRDD<Long, TextWikiPage> dLemmId = dLemmatizedWikiPages
                .mapToPair((page) -> new Tuple2<>(page.getId(), page)); //[ID_PAGE_LEMM, PAGE_LEMM]
        System.out.println(dLemmId.count());

        JavaPairRDD<Long, WikiPage> dDataId = dFullDataset
                .mapToPair((page) -> new Tuple2<>(page.getId(), page)); //[ID_PAGE_ORIG, PAGE_ORIG]
        System.out.println(dDataId.count());

        JavaRDD<WikiPage> map = dDataId
                .join(dLemmId) //[ID_PAGE, {PAGE_ORIG, PAGE_LEMM}]
                .map((pair) -> pair._2._1);//[PAGE_ORIG]

        System.out.println(map.count());

        return map;
    }

    public static JavaRDD<TextWikiPage> filterWikiPageText(JavaRDD<TextWikiPage> dDataset,
                                                           Broadcast<Object2IntOpenHashMap<String>> broadVocabulary,
                                                           int minWordCount) {
        return dDataset.map((page) ->
        {
            Object2IntOpenHashMap<String> lVocabulary = broadVocabulary.getValue();
            ArrayList<String> filteredText = new ArrayList<>();
            for (String word : page.getText()) {
                if (lVocabulary.getInt(word) >= minWordCount) {
                    filteredText.add(word);
                }
            }

            return new TextWikiPage(page.getId(), page.getCategories(), filteredText);
        });
    }


    public JavaRDD<WikiPage> lemmatize(String datasetPath) {
        return Lemmatizer.lemmatizeWikiPages(InputOutput.read(sc, datasetPath));
    }

    public JavaRDD<TextWikiPage> splitWikiPages(JavaRDD<WikiPage> docs) {
        return docs.map((wp) ->
        {
            TextWikiPage newPage = new TextWikiPage();
            newPage.setId(wp.getId());

            ArrayList<String> lemmas = new ArrayList<>();
            lemmas.addAll(Arrays.asList(wp.getText().split(" ")));
            newPage.setText(lemmas);

            newPage.setCategories(new ArrayList<>(Arrays.asList(wp.getCategories())));

            return newPage;
        });
    }

    public JavaRDD<TextWikiPage> loadSplitted(String datasetPath) {
        return splitWikiPages(loadFull(datasetPath));
    }

    public JavaRDD<WikiPage> loadFull(String datasetPath) {
        return InputOutput.read(sc, datasetPath);
    }

    public void save(JavaRDD<WikiPage> dataset, String outputPath) {
        InputOutput.write(dataset, outputPath);
    }

    /**
     * @param pageCat
     * @param bCategoryMap
     * @param k
     * @return
     */
    static ArrayList<String> findTopKCategories(ArrayList<String> pageCat, Broadcast<Object2ObjectOpenHashMap<String, Tuple2<Vector, Integer>>> bCategoryMap, int k) {
        Object2ObjectOpenHashMap<String, Tuple2<Vector, Integer>> frequencyMap = bCategoryMap.getValue();
        ArrayList<String> frequentCategories = new ArrayList<>();
        for (String category : pageCat) {       //insertionSort
            if (frequencyMap.containsKey(category)) {
                if (frequentCategories.isEmpty()) {
                    frequentCategories.add(category);
                } else {
                    for (int i = 0; i < frequentCategories.size(); i++) {
                        if (i == (frequentCategories.size() - 1) && !frequentCategories.contains(category)) {
                            frequentCategories.add(i, category);
                        } else {
                            if (frequencyMap.get(category)._2 > frequencyMap.get(frequentCategories.get(i))._2 && !frequentCategories.contains(category)) {
                                frequentCategories.add(i, category);
                                i = frequentCategories.size() + 1; //to stop the for
                            }
                        }
                    }
                }
            }
        }
        if (frequentCategories.isEmpty())
            frequentCategories.add("");

        if (frequentCategories.size() <= k)
            return frequentCategories;
        else
            return new ArrayList<>(frequentCategories.subList(0, k));
    }

    /**
     * Map each Wikipage to an identical wikipage but with only one category.
     *
     * @param dFullVectorPages The input Dataset.
     * @param bCategoryMap     The HashMap with the frequencies of each word.
     * @return The mapped Dataset.
     */
    JavaRDD<VectorWikiPage> mapCategoriesToFrequent(JavaRDD<VectorWikiPage> dFullVectorPages, Broadcast<Object2ObjectOpenHashMap<String, Tuple2<Vector, Integer>>> bCategoryMap) {
        return dFullVectorPages.map((page) -> {
            Object2ObjectOpenHashMap<String, Tuple2<Vector, Integer>> frequencyMap = bCategoryMap.getValue();
            int max = 0;
            String frequentCategory = "";
            for (String category : page.getCategories()) {    //itero tutte le categorie della wikipage
                if (frequencyMap.containsKey(category)) {
                    if (frequencyMap.get(category)._2 > max) {
                        max = frequencyMap.get(category)._2;
                        frequentCategory = category;
                    }
                }
            }
            ArrayList<String> list = new ArrayList<>();
            list.add(frequentCategory);
            VectorWikiPage mappedVectorWiki = new VectorWikiPage();
            mappedVectorWiki.setId(page.getId());
            mappedVectorWiki.setCategories(list);
            mappedVectorWiki.setVector(page.getVector());
            return mappedVectorWiki;
        })
                .filter((page) -> !page.getCategories().contains(""));
    }

    /**
     * Map each Wikipage to an identical wikipage but with only k categories.
     *
     * @param dWikiset      The input Dataset.
     * @param bCategoryMap The HashMap with the frequencies of each word.
     * @param k             The number of max allowed categories.
     * @return The mapped Dataset.
     */
    JavaRDD<VectorWikiPage> mapCategoriesToKFrequent(JavaRDD<VectorWikiPage> dWikiset,
                                                     Broadcast<Object2ObjectOpenHashMap<String, Tuple2<Vector, Integer>>> bCategoryMap, int k) {
        return dWikiset.map((page) -> {
            VectorWikiPage mappedVectorWiki = new VectorWikiPage();
            mappedVectorWiki.setId(page.getId());
            mappedVectorWiki.setCategories(findTopKCategories(page.getCategories(), bCategoryMap, k));
            mappedVectorWiki.setVector(page.getVector());
            return mappedVectorWiki;
        })
                .filter((page) -> !page.getCategories().contains(""));
    }

    /**
     * For each document of the passed dataset, map the categories to one category. The chosen category will be one
     * of the old categories of the document.
     *
     * @param dVectorPages      input dataset
     * @param bCategoryMap [CATEGORY, CORRESPONDING_VECTOR] map of category to the corresponding vector
     * @return dataset with categories mapped to the categories nearest to the document, the category is chosen from
     * the categories of the document
     */
    public JavaRDD<VectorWikiPage> mapCategoriesToNearest(JavaRDD<VectorWikiPage> dVectorPages, Broadcast<Object2ObjectOpenHashMap<String, Tuple2<Vector, Integer>>> bCategoryMap) {
        return dVectorPages.map((doc) -> {
            double minDistance = Double.MAX_VALUE;
            String nearestCategories = "";
            Object2ObjectOpenHashMap<String, Tuple2<Vector, Integer>> lCategoryMap = bCategoryMap.value();
            for (String cat : doc.getCategories()) {
                if (lCategoryMap.containsKey(cat)) {
                    Vector catVector = lCategoryMap.get(cat)._1;
                    double distance = Distance.distance(catVector, doc.getVector(), distanceType);
                    if (distance < minDistance) {
                        minDistance = distance;
                        nearestCategories = cat;
                    }
                }
            }
            ArrayList<String> newCategory = new ArrayList<>();
            newCategory.add(nearestCategories);
            return new VectorWikiPage(doc.getId(), newCategory, doc.getVector());
        })
                .filter((page) -> !page.getCategories().contains(""));
    }
}
