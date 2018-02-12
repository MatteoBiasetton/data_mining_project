package it.unipd.dei.dm1617;

import edu.stanford.nlp.simple.Document;
import edu.stanford.nlp.simple.Sentence;
import it.unimi.dsi.fastutil.objects.ObjectOpenHashSet;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.broadcast.Broadcast;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.regex.Pattern;


/**
 * Collection of functions that allow to transform texts to sequence
 * of lemmas using lemmatization. An alternative lemmatize is
 * stemming. For a discussion of the difference between stemming and
 * lemmatization see this link: https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html
 */
public class Lemmatizer {

  public static Broadcast<ObjectOpenHashSet<String>> bBannedCategories;
  public static JavaSparkContext sc;

  public static String[] bannedCategoryList = {"birth", "american", "live", "people", "death", "english", "20th-century", "british", "year", "australian", "canadian", "french", "21st-century", "german", "indian", "italian", "19th-century", "fc", "defunct", "japanese", "russian", "scottish", "dutch", "spanish", "swedish", "brazilian", "irish", "2000s", "european", "norwegian", "2010s", "chinese", "belgian", "mexican", "17th-century", "czech", "austrian", "hungarian", "16th-century", "danish", "romanian", "serbian", "turkish", "swiss", "iranian", "portuguese", "finnish", "ukrainian", "st.", "fk", "african", "filipino", "bulgarian", "nigerian", "20th", "15th-century", "13th-century", "albanian", "14th-century", "slovenian", "1940s", "egyptian", "yugoslav", "indonesian", "venezuelan", "12th-century", "estonian", "slovak", "korean", "1990s", "1910s", "11th-century", "belarusian", "st", "dc", "a.c.", "21st", "cb", "kenyan", "moroccan", "4th-century", "19th", "nk", "czechoslovak", "hc", "afc", "dominican", "syrian", "sc", "hebrew", "jamaican", "k", "2nd-century", "1st-century", "7th-century", "9th-century", "macedonian", "kazakhstani", "afghan", "algerian", "palestinian", "j1", "f.c.", "gaelic", "lebanese", "3rd-century", "lok", "ghanaian", "8th-century", "j2", "mm", "paraguayan", "sk"};

  /**
   * Some symbols are interpreted as tokens. This regex allows us to exclude them.
   */
  public static Pattern symbols = Pattern.compile("^[',\\.`/-_&]+$");

  /**
   * A set of special tokens that are present in the Wikipedia dataset
   */
  public static HashSet<String> specialTokens =
          new HashSet<>(Arrays.asList("-lsb-", "-rsb-", "-lrb-", "-rrb-", "'s", "--", "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "a.", "at", "be", "because", "been", "before", "being", "below", "between", "both", "bros.", "but", "by", "can't", "cannot", "could", "couldn't", "did", "des", "di", "ec", "new", "ii", "b", "bc", "ep", "de", "'em", "no.", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "la", "let's", "lo", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"));

  public static void setSc(JavaSparkContext s) {
    sc = s;
  }

  /**
   * Transform a single document in the sequence of its lemmas.
   */
  public static ArrayList<String> lemmatize(String doc) {
    Document d = new Document(doc.toLowerCase());
    // Count spaces to allocate the vector to the right size and avoid trashing memory
    int numSpaces = 0;
    for (int i = 0; i < doc.length(); i++) {
      if (doc.charAt(i) == ' ') {
        numSpaces++;
      }
    }
    ArrayList<String> lemmas = new ArrayList<>(numSpaces);

    for (Sentence sentence : d.sentences()) {
      for (String lemma : sentence.lemmas()) {
        // Remove symbols
        if (!symbols.matcher(lemma).matches() && !specialTokens.contains(lemma)) {
          lemmas.add(lemma);
        }
      }
    }

    return lemmas;
  }

  /**
   * Transform an RDD of strings in the corresponding RDD of lemma
   * sequences, with one sequence for each original document.
   */
  public static JavaRDD<ArrayList<String>> lemmatize(JavaRDD<String> docs) {
    return docs.map((d) -> lemmatize(d));
  }

  /**
   * Transform an RDD of WikiPage objects into an RDD of WikiPage
   * objects with the text replaced by the concatenation of the lemmas
   * in each page.
   */
  public static JavaRDD<WikiPage> lemmatizeWikiPages(JavaRDD<WikiPage> docs) {
    return docs.map((wp) -> {
      ArrayList<String> lemmas = lemmatize(wp.getText());
      StringBuilder newText = new StringBuilder();
      for (String lemma : lemmas)
        newText.append(lemma).append(' ');
      wp.setText(newText.toString());

      ArrayList<String> newCategories = new ArrayList<>();

      for (String c : wp.getCategories()) {
          if (!lemmatize(c).isEmpty()) {
              String lemmatized = lemmatize(c).get(0);

              boolean isLemmatizable = !(lemmatized == null || lemmatized.equals(""));
              boolean isAdmissible = !bBannedCategories.value().contains(lemmatized);

              //If the category could be lemmatized and is not in the banned set
              if (isLemmatizable && isAdmissible)
                  newCategories.add(lemmatized);
          }
      }

      Object[] l = newCategories.toArray();
      String[] d = new String[l.length];
      System.arraycopy(l, 0, d, 0, l.length);

      wp.setCategories(d);
      return wp;
    })
            .filter((wikipage) -> wikipage.getCategories().length > 0);
  }

  public static void generateBannedCategorySet() {
    ObjectOpenHashSet<String> lBannedCategories = new ObjectOpenHashSet<>();
    lBannedCategories.addAll(Arrays.asList(bannedCategoryList));
    System.out.println(lBannedCategories.size());
    bBannedCategories = sc.broadcast(lBannedCategories);
  }

  public static void main(String[] args) {
    System.out.println(lemmatize("This is a sentence. This is another. The whole thing is a document made of sentences. gone go went."));
  }

}
