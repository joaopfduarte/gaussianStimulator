// ... existing code ...
import converter.IrisConverter;
import converter.Register;
import converter.VertebralColumnConverter;

import java.util.List;

public class GaussianMain {
    public static void main(String[] args) {
        // true = Iris, false = Vertebral
        boolean usarIris = false;

        IrisConverter iris = null;
        VertebralColumnConverter vertebral = null;

        if (usarIris) {
            iris = new IrisConverter();
        } else {
            vertebral = new VertebralColumnConverter();
        }

        // Carrega treino e teste
        List<Register> trainList = usarIris ? iris.trainConverter() : vertebral.trainConverter();
        List<Register> testList  = usarIris ? iris.testConverter()  : vertebral.testConverter();

        // Converte para matrizes esperadas pelo GaussianImp
        double[][] Xtrain = new double[trainList.size()][];
        String[]   ytrain = new String[trainList.size()];
        for (int i = 0; i < trainList.size(); i++) {
            Register r = trainList.get(i);
            Xtrain[i] = r.getFeatures();
            ytrain[i] = targetToLabel(r.getTarget(), usarIris);
        }

        double[][] Xtest = new double[testList.size()][];
        String[]   ytest = new String[testList.size()];
        for (int i = 0; i < testList.size(); i++) {
            Register r = testList.get(i);
            Xtest[i] = r.getFeatures();
            ytest[i] = targetToLabel(r.getTarget(), usarIris);
        }

        // Treina
        GaussianImp clf = new GaussianImp();
        clf.train(Xtrain, ytrain);

        // Avalia e imprime resultados organizados
        System.out.println("=== RESULTADOS TREINO ===");
        int errosTreino = imprimirErrosPorAmostra(clf, Xtrain, ytrain);
        double accTrain = clf.evaluate(Xtrain, ytrain);
        System.out.println("Soma de erros (treino): " + errosTreino);
        System.out.println("Acuracia (treino): " + String.format(java.util.Locale.US, "%.4f", accTrain));

        System.out.println("\n=== RESULTADOS TESTE ===");
        int errosTeste = imprimirErrosPorAmostra(clf, Xtest, ytest);
        double accTest = clf.evaluate(Xtest, ytest);
        System.out.println("Soma de erros (teste): " + errosTeste);
        System.out.println("Acuracia (teste): " + String.format(java.util.Locale.US, "%.4f", accTest));
    }

    // Converte o vetor alvo em rótulo String para o GaussianImp
    private static String targetToLabel(double[] target, boolean usarIris) {
        // Nos conversores atuais, target[0] contém a classe codificada
        int v = (int) Math.round(target[0]);
        if (usarIris) {
            // 0 -> Iris-setosa, 1 -> Iris-versicolor, 2 -> Iris-virginica
            switch (v) {
                case 0: return "Iris-setosa";
                case 1: return "Iris-versicolor";
                case 2: return "Iris-virginica";
                default: return "UNKNOWN";
            }
        } else {
            // Vertebral: 1.0 para AB, 0.0 para NO
            return v == 1 ? "AB" : "NO";
        }
    }

    // Imprime predição por amostra, erro e acumula soma de erros
    private static int imprimirErrosPorAmostra(GaussianImp clf, double[][] X, String[] y) {
        int somaErros = 0;
        for (int i = 0; i < X.length; i++) {
            String pred = clf.predict(X[i]);
            int erro = (pred != null && pred.equals(y[i])) ? 0 : 1;
            somaErros += erro;

            System.out.println(
                    String.format(java.util.Locale.US,
                            "Amostra %d | Verdadeiro: %-15s | Predito: %-15s | Erro: %d",
                            i, y[i], pred, erro
                    )
            );
        }
        return somaErros;
    }
}