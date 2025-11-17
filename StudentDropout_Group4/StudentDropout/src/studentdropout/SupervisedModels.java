package studentdropout;

import weka.core.Instances;
import weka.core.Attribute;
import weka.classifiers.Classifier;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.J48;
import weka.classifiers.functions.SMO;
import weka.classifiers.Evaluation;
import weka.attributeSelection.InfoGainAttributeEval;
import weka.attributeSelection.Ranker;
import weka.attributeSelection.AttributeSelection;

import java.util.Random;
import weka.filters.Filter;

public class SupervisedModels {

    // Train & evaluate all classifiers (multiclass)
    public static void trainAndEvaluateAll(Instances data) throws Exception {
        System.out.println("\n--- MULTI-CLASS EXPERIMENTS ---");

        if (!hasMultipleClasses(data)) {
            System.out.println("Not enough classes to train classifiers.");
            return;
        }

        Classifier[] classifiers = new Classifier[]{
                new RandomForest(),
                new J48(),
                new SMO()
        };

        for (Classifier cls : classifiers) {
            System.out.println("\nTraining: " + cls.getClass().getSimpleName());
            cls.buildClassifier(data);
            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(cls, data, 10, new Random(1));
            System.out.println(eval.toSummaryString());
            System.out.println(eval.toClassDetailsString()); // Precision, Recall, F1
            System.out.println(eval.toMatrixString());       // Confusion Matrix
            System.out.printf("ROC AUC (macro): %.4f%n", averageROC(eval, data));
        }

        printFeatureImportance(data);
    }

    // Train & evaluate for binary dataset
    public static void trainAndEvaluateBinary(Instances data) throws Exception {
        System.out.println("\n--- BINARY DROP-OUT EXPERIMENTS ---");

        if (!hasMultipleClasses(data)) {
            System.out.println("Not enough classes to train classifiers.");
            return;
        }

        Classifier[] classifiers = new Classifier[]{
                new RandomForest(),
                new J48(),
                new SMO()
        };

        for (Classifier cls : classifiers) {
            System.out.println("\nTraining: " + cls.getClass().getSimpleName());
            cls.buildClassifier(data);
            Evaluation eval = new Evaluation(data);
            eval.crossValidateModel(cls, data, 10, new Random(1));
            System.out.println(eval.toSummaryString());
        }

        printFeatureImportance(data);
    }

    // --- Feature importance using InfoGain ---
    public static void printFeatureImportance(Instances data) {
        try {
            Attribute classAttr = data.classAttribute();
            if (!classAttr.isNominal()) {
                System.out.println("Class attribute is not nominal. Skipping InfoGain.");
                return;
            }

            InfoGainAttributeEval eval = new InfoGainAttributeEval();
            eval.buildEvaluator(data);

            System.out.println("\n=== Feature Importance (InfoGain) ===");
            for (int i = 0; i < data.numAttributes(); i++) {
                if (i == data.classIndex()) continue;
                double score = eval.evaluateAttribute(i);
                System.out.printf("%s : %.4f%n", data.attribute(i).name(), score);
            }

        } catch (Exception e) {
            System.out.println("Error computing InfoGain: " + e.getMessage());
        }
    }

    // --- Helper: check if dataset has more than one class value ---
    private static boolean hasMultipleClasses(Instances data) {
        if (data.classAttribute() == null) return false;
        return data.classAttribute().numValues() > 1;
    }
    
    private static double averageROC(Evaluation eval, Instances data) throws Exception {
        double sum = 0.0;
        int classes = data.numClasses();
        for (int i = 0; i < classes; i++) {
            sum += eval.areaUnderROC(i);
        }
        return sum / classes;
    }

}
