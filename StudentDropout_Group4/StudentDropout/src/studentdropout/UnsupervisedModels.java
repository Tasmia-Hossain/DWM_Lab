package studentdropout;

import weka.clusterers.SimpleKMeans;
import weka.clusterers.EM;
import weka.core.Instances;
import weka.core.Instance;

import java.util.HashMap;
import weka.core.Attribute;

public class UnsupervisedModels {

    public static void runKMeans(Instances data, int k) throws Exception {
        Instances clusterData = new Instances(data);
        clusterData.setClassIndex(-1);

        SimpleKMeans kmeans = new SimpleKMeans();
        kmeans.setNumClusters(k);
        kmeans.setPreserveInstancesOrder(true);
        kmeans.setSeed(10);
        kmeans.buildClusterer(clusterData);

        int[] assignments = kmeans.getAssignments();
        printClusterProfiles(assignments, data);
    }

    public static void runEM(Instances data) throws Exception {
        Instances clusterData = new Instances(data);
        clusterData.setClassIndex(-1);

        EM em = new EM();
        em.setSeed(10);
        em.buildClusterer(clusterData);

        int[] assignments = new int[clusterData.numInstances()];
        for (int i = 0; i < clusterData.numInstances(); i++) assignments[i] = em.clusterInstance(clusterData.instance(i));

        printClusterProfiles(assignments, data);
    }

    private static void printClusterProfiles(int[] assignments, Instances originalData) {
    HashMap<Integer, HashMap<String, Integer>> clusterClassCounts = new HashMap<>();
    HashMap<Integer, Integer> clusterSizes = new HashMap<>();

    // Count class occurrences per cluster
    for (int i = 0; i < assignments.length; i++) {
        int cluster = assignments[i];
        clusterSizes.put(cluster, clusterSizes.getOrDefault(cluster, 0) + 1);

        String cls = originalData.instance(i).stringValue(originalData.classIndex());
        clusterClassCounts.putIfAbsent(cluster, new HashMap<>());
        HashMap<String, Integer> map = clusterClassCounts.get(cluster);
        map.put(cls, map.getOrDefault(cls, 0) + 1);
    }

    System.out.println("\n=== Cluster Profiles ===");
    for (int c : clusterClassCounts.keySet()) {
        int clusterSize = clusterSizes.get(c);
        System.out.println("Cluster " + c + " (size: " + clusterSize + "):");

        // Class distribution
        System.out.print("  Class distribution: ");
        for (String cls : clusterClassCounts.get(c).keySet()) {
            int cnt = clusterClassCounts.get(c).get(cls);
            System.out.printf("%s=%.2f%% ", cls, (cnt * 100.0) / clusterSize);
        }
        System.out.println();

        // Numeric attribute summary: mean & std dev
        System.out.println("  Numeric attribute summary:");
        for (int a = 0; a < originalData.numAttributes(); a++) {
            Attribute attr = originalData.attribute(a);
            if (attr.isNumeric()) {
                double sum = 0.0, sumSq = 0.0, count = 0;
                for (int i = 0; i < assignments.length; i++) {
                    if (assignments[i] == c) {
                        double val = originalData.instance(i).value(a);
                        sum += val;
                        sumSq += val * val;
                        count++;
                    }
                }
                if (count > 0) {
                    double mean = sum / count;
                    double std = Math.sqrt(sumSq / count - mean * mean);
                    System.out.printf("    %s: mean=%.4f std=%.4f%n", attr.name(), mean, std);
                }
            }
        }
        System.out.println();
    }
}

}
