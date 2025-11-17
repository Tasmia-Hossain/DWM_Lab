package studentdropout;

import weka.core.Instances;

import java.util.HashMap;

public class ExploratoryAnalyzer {

    public static void printBasicStats(Instances data) {
        System.out.println("\n=== Exploratory Analysis ===");
        System.out.println("Number of instances: " + data.numInstances());
        System.out.println("Number of attributes: " + data.numAttributes());

        for (int i = 0; i < data.numAttributes(); i++) {
            if (data.attribute(i).isNumeric()) {
                double sum = 0, sumsq = 0;
                for (int j = 0; j < data.numInstances(); j++) {
                    double val = data.instance(j).value(i);
                    sum += val;
                    sumsq += val * val;
                }
                double mean = sum / data.numInstances();
                double variance = (sumsq / data.numInstances()) - (mean * mean);
                double std = Math.sqrt(Math.max(0.0, variance));
                System.out.printf("Attribute %s - mean: %.3f, std: %.3f\n",
                        data.attribute(i).name(), mean, std);
            }
        }
    }

    public static void printClassDistribution(Instances data) {
        if (data.classIndex() < 0) {
            System.out.println("Dataset has no class index set.");
            return;
        }
        HashMap<String, Integer> counts = new HashMap<>();
        for (int i = 0; i < data.numInstances(); i++) {
            String cls = data.instance(i).stringValue(data.classIndex());
            counts.put(cls, counts.getOrDefault(cls, 0) + 1);
        }
        System.out.println("\nClass distribution:");
        for (String k : counts.keySet()) {
            System.out.println(k + ": " + counts.get(k));
        }
    }
}
