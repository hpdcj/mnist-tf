package pl.icm.lgorski;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.apache.commons.lang3.ArrayUtils;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.io.*;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

//cf. https://github.com/tensorflow/models/blob/master/samples/languages/java/training/src/main/java/Train.java
public class Test {

    private static final int BATCH_SIZE = 100;
    private static final int IMAGE_SIZE = 28*28;

    @RequiredArgsConstructor
    private static class MnistImageBatch {
        @Getter
        private final Tensor<Integer> label;
        @Getter
        private final Tensor<Float> pixels;

    }


    private static List<MnistImageBatch> readMnistImages (List<String> lines) {
        var batches = new ArrayList<MnistImageBatch>();
        var currentBatch = new ArrayList<String>(BATCH_SIZE);
        for (int i = 0; i < lines.size(); i++) {
            currentBatch.add(lines.get(i));
            if (currentBatch.size() % BATCH_SIZE == 0) {
                addProcessedLineToBatch(batches, currentBatch);
                currentBatch.clear();
            }
        }
        if (!currentBatch.isEmpty()) {
            addProcessedLineToBatch(batches, currentBatch);
        }
        return batches;
    }

    private static void addProcessedLineToBatch(List<MnistImageBatch> batches, ArrayList<String> currentBatch) {
        var batch = processBatch(currentBatch);
        batches.add(batch);
    }

    private static MnistImageBatch processBatch(List<String> currentBatch) {
        int numberOfElements = currentBatch.size();
        IntBuffer ys = IntBuffer.allocate(numberOfElements);
        FloatBuffer xs = FloatBuffer.allocate(numberOfElements * IMAGE_SIZE);

        currentBatch.stream()
                .forEach( line -> {
                    var split = line.split(" ");
                    ys.put(Integer.valueOf(split[0]));
                    Arrays.stream(split).skip(1).map(Float::valueOf).forEach(xs::put);
                });

        ys.flip();
        xs.flip();

        var tensorYs = Tensor.create(new long[] { numberOfElements }, ys);
        var tensorXs = Tensor.create(new long[] { numberOfElements, IMAGE_SIZE}, xs);
        return new MnistImageBatch(tensorYs, tensorXs);
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
            sess.runner().addTarget("init").run();
            sess.runner().feed("X", trainImages.get(0).pixels).feed("y", trainImages.get(0).label).addTarget("train/optimize").run();
        }

    }
}
