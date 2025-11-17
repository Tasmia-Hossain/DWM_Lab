package studentdropout;

import weka.associations.Apriori;
import weka.associations.FPGrowth;
import weka.associations.AssociationRule;
import weka.associations.AssociationRules;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;

import java.util.List;

public class AssociationMiner {

    public static void runApriori(Instances data, double minSupport, double minConfidence, int numRules) throws Exception {
        System.out.println("\n=== Apriori Analysis ===\n");

        // Discretize numeric attributes
        Discretize disc = new Discretize();
        disc.setInputFormat(data);
        Instances discData = Filter.useFilter(data, disc);

        Apriori apriori = new Apriori();
        apriori.setLowerBoundMinSupport(minSupport);
        apriori.setMinMetric(minConfidence);
        apriori.setNumRules(numRules);
        apriori.buildAssociations(discData);

        System.out.printf("Minimum support: %.2f (%d instances)%n", minSupport, 
                (int)(minSupport * data.numInstances()));
        System.out.printf("Minimum metric <confidence>: %.2f%n", minConfidence);

        // Get rules safely
        AssociationRules rules = apriori.getAssociationRules();
        List<AssociationRule> ruleList = rules.getRules();

        System.out.println("\nBest rules found:\n");
        int count = 1;
        for (AssociationRule r : ruleList) {
            if (count > numRules) break;

            double conf = r.getPrimaryMetricValue(); // safe
            if (conf >= minConfidence) {
                // Print only premise, consequence, support, confidence
                System.out.printf("%2d. %s ==> %s (%d) <conf: %.2f>\n",
                        count,
                        r.getPremise(),
                        r.getConsequence(),
                        r.getTotalSupport(),
                        conf
                );
                count++;
            }
        }
    }
}
