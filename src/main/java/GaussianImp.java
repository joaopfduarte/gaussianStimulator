import java.util.*;

public class GaussianImp {
   private Map<String, Double> classPriors;
   private Map<String, double[]> classMeans;
   private Map<String, double[]> classStdDevs;
   private Map<String, List<double[]>> trainingData;

   private int numFeatures;
   private Set<String> classLabels;

   public GaussianImp() {
       this.classPriors = new HashMap<>();
       this.classMeans = new HashMap<>();
       this.classStdDevs = new HashMap<>();
       this.trainingData = new HashMap<>();
       this.classLabels = new HashSet<>();
   }

   public void train(double[][] features, String[] labels) {
       this.numFeatures = features[0].length;

       for (int i = 0; i < features.length; i++) {
           String label = labels[i];
           classLabels.add(label);

           trainingData.computeIfAbsent(label, k -> new ArrayList<>()).add(features[i]);
       }

       for (String classLabel : classLabels) {
           List<double[]> classData = trainingData.get(classLabel);
           int classSize = classData.size();

           classPriors.put(classLabel, (double) classSize / features.length);

           double[] means = new double[numFeatures];
           double[] stdDevs = new double[numFeatures];

           for (int feature = 0; feature < numFeatures; feature++) {
               double sum = 0.0;
               for (double[] sample : classData) {
                   double diff = sample[feature] - means[feature];
                   sum += diff * diff;
               }
               stdDevs[feature] = Math.sqrt(sum / classSize);

               if(stdDevs[feature] == 0) {
                   stdDevs[feature] = 1e-6;
               }
           }

           classMeans.put(classLabel, means);
           classStdDevs.put(classLabel, stdDevs);
       }
   }

   private double gaussianProbability(double x, double mean, double stdDev) {
       double coefficient = 1.0 / (Math.sqrt(2 * Math.PI) * stdDev);
       double expoent = -0.5 * Math.pow((x - mean) / stdDev, 2);
        return coefficient * Math.exp(expoent);
   }

   public Map<String, Double> classify(double[] features) {
       Map<String, Double> classProbabilities = new HashMap<>();

       for (String classLabel : classLabels) {
           double[] means = classMeans.get(classLabel);
           double[] stdDevs = classStdDevs.get(classLabel);
           double prior = classPriors.get(classLabel);

           double likelihood = 1.0;
           for (int k = 0; k < numFeatures; k++) {
               double prob = gaussianProbability(features[k], means[k], stdDevs[k]);
               likelihood *= prob;
           }

           double posterior = prior * likelihood;
           classProbabilities.put(classLabel, posterior);
       }

       double total = classProbabilities.values().stream().mapToDouble(Double::doubleValue).sum();
       if (total > 0) {
           classProbabilities.replaceAll((k,v) -> v / total);
       }

       return classProbabilities;
   }

   public String predict(double[] features) {
       Map<String, Double> probabilities = classify(features);
       return probabilities.entrySet().stream()
               .max(Map.Entry.comparingByValue())
               .map(Map.Entry::getKey)
               .orElse(null);
   }

   public double evaluate(double[][] testFeatures, String[] testLabels) {
       int correct = 0;
       for (int i = 0; i < testFeatures.length; i++) {
           String predicted = predict(testFeatures[i]);
           if (predicted != null && predicted.equals(testLabels[i])) {
               correct++;
           }
       }
       return (double) correct / testFeatures.length;
   }

    public Map<String, Double> getClassPriors() { return classPriors; }
    public Map<String, double[]> getClassMeans() { return classMeans; }
    public Map<String, double[]> getClassStdDevs() { return classStdDevs; }
}
