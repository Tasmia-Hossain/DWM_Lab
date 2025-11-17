package studentdropout;

import weka.core.Attribute;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.*;
import java.io.File;
import java.util.ArrayList;

public class DataPreprocessor {

    // Load CSV and clean attribute names
    public static Instances loadAndCleanCSV(String path) throws Exception {
        CSVLoader loader = new CSVLoader();
        loader.setSource(new File(path));
        Instances data = loader.getDataSet();

        for (int i = 0; i < data.numAttributes(); i++) {
            String name = data.attribute(i).name().trim().replaceAll("\\s+", "_");
            data.renameAttribute(i, name);
        }
        return data;
    }

    // Preprocess all attributes
    public static Instances preprocess(Instances raw) throws Exception {
        Instances data = new Instances(raw);

        // 1. Replace missing values
        ReplaceMissingValues replace = new ReplaceMissingValues();
        replace.setInputFormat(data);
        data = Filter.useFilter(data, replace);

        // 2. Convert string attributes to nominal
        StringToNominal str2nom = new StringToNominal();
        str2nom.setAttributeRange("first-last");
        str2nom.setInputFormat(data);
        data = Filter.useFilter(data, str2nom);

        // 3. Feature engineering
        addTotalCredits(data);
        addPassRateAndEvalLoad(data);
        addFinancialRisk(data);

        // 4. Detect class attribute (nominal with >1 value)
        Attribute classAttr = findClassAttribute(data);
        if (classAttr == null) throw new Exception("No suitable class attribute found!");

        // 5. Convert class to nominal if needed
        if (classAttr.isNumeric()) {
            NumericToNominal num2nom = new NumericToNominal();
            num2nom.setAttributeIndices("" + (classAttr.index() + 1));
            num2nom.setInputFormat(data);
            data = Filter.useFilter(data, num2nom);
            classAttr = data.attribute(classAttr.name());
        }

        data.setClass(classAttr);

        // 6. Standardize numeric attributes
        Standardize standardize = new Standardize();
        standardize.setIgnoreClass(true);
        standardize.setInputFormat(data);
        data = Filter.useFilter(data, standardize);

        return data;
    }

    // --- Feature Engineering Helpers ---
    private static void addTotalCredits(Instances data) throws Exception {
        Attribute cred1 = data.attribute("Curricular_units_1st_sem_(credited)");
        Attribute cred2 = data.attribute("Curricular_units_2nd_sem_(credited)");
        if (cred1 != null && cred2 != null) {
            AddExpression expr = new AddExpression();
            expr.setExpression("a" + (cred1.index() + 1) + " + a" + (cred2.index() + 1));
            expr.setName("Total_credits");
            expr.setInputFormat(data);
            data = Filter.useFilter(data, expr);
        }
    }

    private static void addPassRateAndEvalLoad(Instances data) {
        Attribute approved = data.attribute("Curricular_units_1st_sem_(approved)");
        Attribute enrolled = data.attribute("Curricular_units_1st_sem_(enrolled)");
        Attribute evals = data.attribute("Curricular_units_1st_sem_(evaluations)");
        if (approved != null && enrolled != null) {
            Attribute passRate = new Attribute("pass_rate_1st_sem");
            data.insertAttributeAt(passRate, data.numAttributes());
            int newIdx = data.attribute("pass_rate_1st_sem").index();
            for (int i = 0; i < data.numInstances(); i++) {
                double rate = (data.instance(i).value(enrolled) > 0)
                        ? data.instance(i).value(approved) / data.instance(i).value(enrolled)
                        : 0.0;
                data.instance(i).setValue(newIdx, rate);
            }

            if (evals != null) {
                Attribute load = new Attribute("eval_load_1st_sem");
                data.insertAttributeAt(load, data.numAttributes());
                newIdx = data.attribute("eval_load_1st_sem").index();
                for (int i = 0; i < data.numInstances(); i++) {
                    double l = (data.instance(i).value(enrolled) > 0)
                            ? data.instance(i).value(evals) / data.instance(i).value(enrolled)
                            : 0.0;
                    data.instance(i).setValue(newIdx, l);
                }
            }
        }
    }

    private static void addFinancialRisk(Instances data) {
        Attribute debtor = data.attribute("Debtor");
        Attribute tuition = data.attribute("Tuition_fees_up_to_date");
        Attribute finRisk = new Attribute("financial_risk");
        data.insertAttributeAt(finRisk, data.numAttributes());
        int idx = data.attribute("financial_risk").index();
        for (int i = 0; i < data.numInstances(); i++) {
            double risk = 0.0;
            try {
                if (debtor != null && data.instance(i).stringValue(debtor).equalsIgnoreCase("yes")) risk = 1.0;
                if (tuition != null && data.instance(i).stringValue(tuition).equalsIgnoreCase("no")) risk = 1.0;
            } catch (Exception e) {}
            data.instance(i).setValue(idx, risk);
        }
    }

    private static Attribute findClassAttribute(Instances data) {
        for (int i = 0; i < data.numAttributes(); i++) {
            Attribute attr = data.attribute(i);
            if (attr.isNominal() && attr.numValues() > 1) return attr;
        }
        return null;
    }

    // --- Binary dataset with leakage removal ---
    public static Instances createBinaryDataset(Instances multiclassData) throws Exception {
        Instances data = new Instances(multiclassData);
        ArrayList<String> labels = new ArrayList<>();
        labels.add("NonDropout");
        labels.add("Dropout");

        Attribute binAttr = new Attribute("dropout_binary", labels);
        data.insertAttributeAt(binAttr, data.numAttributes());
        int newIdx = data.attribute("dropout_binary").index();
        int oldClassIdx = data.classIndex();

        for (int i = 0; i < data.numInstances(); i++) {
            String origClass = data.instance(i).stringValue(oldClassIdx);
            data.instance(i).setValue(newIdx, origClass.equalsIgnoreCase("Dropout") ? "Dropout" : "NonDropout");
        }

        data.setClassIndex(newIdx);

        // 🚨 Remove leakage attributes that perfectly reveal dropout
        String[] leakAttrs = {
            "Curricular_units_1st_sem_(approved)",
            "Curricular_units_1st_sem_(grade)",
            "Curricular_units_1st_sem_(evaluations)",
            "Curricular_units_2nd_sem_(approved)",
            "Curricular_units_2nd_sem_(grade)",
            "Curricular_units_2nd_sem_(evaluations)",
            "pass_rate_1st_sem",
            "eval_load_1st_sem",
            "financial_risk",
            "Target"  // direct outcome variable
        };

        for (String name : leakAttrs) {
            Attribute attr = data.attribute(name);
            if (attr != null) {
                System.out.println("Removing leakage attribute: " + name);
                data.deleteAttributeAt(attr.index());
            }
        }

        return data;
    }
}
