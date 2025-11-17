package studentdropout;

import weka.core.Instances;
import weka.classifiers.trees.RandomForest;

public class StudentDropout {

    public static void main(String[] args) {
        try {
            String csvPath = "Data/student_dropout.csv";

            Instances raw = DataPreprocessor.loadAndCleanCSV(csvPath);
            Instances pre = DataPreprocessor.preprocess(raw);

            //ExploratoryAnalyzer.printBasicStats(pre);
            ExploratoryAnalyzer.printClassDistribution(pre);

            // Multiclass training
            SupervisedModels.trainAndEvaluateAll(pre);

            // Binary dropout dataset
            Instances bin = DataPreprocessor.createBinaryDataset(pre);
            ExploratoryAnalyzer.printClassDistribution(bin);
            SupervisedModels.trainAndEvaluateBinary(bin);

            // Clustering
            UnsupervisedModels.runKMeans(pre, 4);
            UnsupervisedModels.runEM(pre);

            // Association rules
            AssociationMiner.runApriori(pre, 0.05, 0.7, 20);

            // Save model
            RandomForest rf = new RandomForest();
            rf.setNumIterations(100);
            rf.setSeed(1);
            rf.buildClassifier(pre);
            ModelPersistence.saveModel(rf, "models/RandomForest_full.model");

            ModelPersistence.loadModel("models/RandomForest_full.model");

            System.out.println("\n=== Pipeline Complete ===");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
