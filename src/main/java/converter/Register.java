package converter;

public class Register {
    private final double[] features;
    private final double[] target;

    public Register(double[] features, double[] target) {
        this.features = features;
        this.target = target;
    }

    public double[] getFeatures() {
        return features;
    }

    public double[] getTarget() {
        return target;
    }
}
