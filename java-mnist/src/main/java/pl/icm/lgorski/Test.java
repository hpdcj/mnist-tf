package pl.icm.lgorski;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.apache.commons.lang3.ArrayUtils;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.io.*;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.StreamSupport;

//cf. https://github.com/tensorflow/models/blob/master/samples/languages/java/training/src/main/java/Train.java
public class Test {

    @RequiredArgsConstructor
    private static class MnistImage {
        @Getter
        private final int label;
        @Getter
        private final Tensor<Float> pixels;

    }

    private static List<MnistImage> readMnistImages (List<String> lines) {
        double[] arr = null;
        return lines.stream()
                .map(string -> string.split(" "))
                .map(array -> Arrays.stream(array).map(Float::valueOf).toArray(Float[]::new))
                .map(ArrayUtils::toPrimitive)
                .map( array -> new MnistImage((int)array[0],
                                                  Tensor.create(new long[]{array.length - 1}, FloatBuffer.wrap(array, 1, array.length - 1))))
                .collect(Collectors.toList());

    }
    private static void showOperations (Graph graph) {
        var iter = graph.operations();
        while (iter.hasNext()) {
            System.out.println(""+iter.next());
        }
    }

    public static void main (String[] args) throws IOException {
        final byte[] graphDef = Files.readAllBytes(Paths.get("../graph.pb"));
        final var trainImages = readMnistImages(Files.readAllLines(Paths.get("../mnist.train.txt")));
        final var testImages = readMnistImages(Files.readAllLines(Paths.get("../mnist.test.txt")));

        try (Graph graph = new Graph();
        Session sess = new Session((graph))) {
            graph.importGraphDef(graphDef);
            showOperations(graph);
        }

    }
}
