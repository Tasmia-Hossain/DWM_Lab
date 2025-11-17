package studentdropout;

import weka.classifiers.Classifier;
import weka.core.SerializationHelper;

import java.io.File;

public class ModelPersistence {

    public static void saveModel(Classifier model, String path) throws Exception {
        File folder = new File(path).getParentFile();
        if (!folder.exists()) folder.mkdirs();
        SerializationHelper.write(path, model);
        System.out.println("Model saved to: " + path);
    }

    public static Classifier loadModel(String path) throws Exception {
        File f = new File(path);
        if (!f.exists()) {
            System.out.println("Model file not found: " + f.getAbsolutePath());
            return null;
        }
        Object obj = SerializationHelper.read(f.getAbsolutePath());
        if (obj instanceof Classifier) {
            System.out.println("Model loaded: " + ((Classifier) obj).getClass().getSimpleName());
            return (Classifier) obj;
        } else {
            System.out.println("Deserialized object is not a Classifier: " + obj.getClass().getName());
            return null;
        }
    }
}
