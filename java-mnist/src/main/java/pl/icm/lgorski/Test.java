package pl.icm.lgorski;

import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.stream.StreamSupport;

//cf. https://github.com/tensorflow/models/blob/master/samples/languages/java/training/src/main/java/Train.java
public class Test {


    private static class MnistImage {

    }
    private static void showOperations (Graph graph) {
        var iter = graph.operations();
        while (iter.hasNext()) {
            System.out.println(""+iter.next());
        }
    }

    public static void main (String[] args) throws IOException {
        final byte[] graphDef = Files.readAllBytes(Paths.get("../graph.pb"));
        try (Graph graph = new Graph();
        Session sess = new Session((graph))) {
            graph.importGraphDef(graphDef);
            showOperations(graph);
        }

    }
}
