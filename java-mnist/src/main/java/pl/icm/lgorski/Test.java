package pl.icm.lgorski;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.apache.commons.collections4.ListUtils;
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
import java.util.Collections;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

//cf. https://github.com/tensorflow/models/blob/master/samples/languages/java/training/src/main/java/Train.java
public class Test {

    private static final int BATCH_SIZE = 100;
    private static final int IMAGE_SIZE = 28*28;

    @RequiredArgsConstructor
    private static class MnistImage {
        @Getter
        private final int label;
        @Getter
        private final float[] pixels;
    }

    @RequiredArgsConstructor
    private static class MnistImageBatch {
        @Getter
        private final Tensor<Integer> label;
        @Getter
        private final Tensor<Float> pixels;

    }

    private static class LayerWeights {
        @Getter
        private final int dimX;
        @Getter
        private final int dimY;

        @Getter
        private final float[][] weights;

        public LayerWeights(int dimX, int dimY) {
            this.dimX = dimX;
            this.dimY = dimY;
            weights = new float[dimX][dimY];
        }
    }
    private static List<MnistImage> readMnistImages (List<String> lines) {
        return lines.stream()
                .map(line -> line.split(" "))
                .map( numbers -> {
                    var label = Integer.parseInt(numbers[0]);
                    var pixels = Arrays.stream(numbers).skip(1).map(Float::valueOf).toArray(Float[]::new);
                    return new MnistImage(label, ArrayUtils.toPrimitive(pixels));
                })
                .collect(Collectors.toList());
    }

    private static List<MnistImageBatch> batchMnist (List<MnistImage> images, int batchSize) {
        return ListUtils.partition(images, batchSize)
                .stream()
                .map(Test::processBatch)
                .collect(Collectors.toList());
    }

    private static MnistImageBatch processBatch(List<MnistImage> currentBatch) {
        int numberOfElements = currentBatch.size();
        IntBuffer ys = IntBuffer.allocate(numberOfElements);
        FloatBuffer xs = FloatBuffer.allocate(numberOfElements * IMAGE_SIZE);

        currentBatch.stream()
                .forEach( image -> {
                    ys.put(image.getLabel());
                    xs.put(image.getPixels());
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

        var trainImagesBatches = batchMnist(trainImages, BATCH_SIZE);
        Collections.shuffle(trainImagesBatches);
        final var testImagesBatches = batchMnist(testImages, 1);

        try (Graph graph = new Graph();
        Session sess = new Session((graph))) {
            graph.importGraphDef(graphDef);
            sess.runner().addTarget("init").run();

            var start = System.nanoTime();
            testImagesBatches.forEach( batch -> {
                sess.runner().feed("X", batch.getPixels())
                        .feed("y", batch.getLabel())
                        .addTarget("train/optimize")
                        .run();
                var hidden1Weights = sess.runner().fetch("hidden1/weights").run();
                LayerWeights hidden1WeightsArray = new LayerWeights(28*28, 300);
                hidden1Weights.forEach( tensor -> {
                    tensor.copyTo(hidden1WeightsArray.getWeights());
                });
                var hidden1Biases = sess.runner().fetch("hidden1/biases").run();
                LayerWeights hidden1BiasesArray = new LayerWeights(1, 300);
                hidden1Biases.forEach( tensor -> {
                    tensor.copyTo(hidden1BiasesArray.getWeights()[0]);
                });
                System.out.println("END");
            });
            var stop = System.nanoTime();
            System.out.println("Time = "  + (stop - start)*1e-9);
        }
    }
}
