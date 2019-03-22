package pl.icm.lgorski;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.apache.commons.collections4.ListUtils;
import org.apache.commons.lang3.ArrayUtils;
import org.pcj.*;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

//cf. https://github.com/tensorflow/models/blob/master/samples/languages/java/training/src/main/java/Train.java

@RegisterStorage(MnistGradientDescent.Shared.class)
public class MnistGradientDescent implements StartPoint {

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
        private final FloatBuffer weights;

        @Getter
        private final String layerName;

        public LayerWeights(String layerName, int size){
            weights = FloatBuffer.allocate(size);
            this.layerName = layerName;
        }
    }

    private List<MnistImage> readMnistImages (List<String> lines) {
        return lines.stream()
                .map(line -> line.split(" "))
                .map( numbers -> {
                    var label = Integer.parseInt(numbers[0]);
                    var pixels = Arrays.stream(numbers).skip(1).map(Float::valueOf).toArray(Float[]::new);
                    return new MnistImage(label, ArrayUtils.toPrimitive(pixels));
                })
                .collect(Collectors.toList());
    }

    private List<MnistImageBatch> batchMnist (List<MnistImage> images, int batchSize) {
        return ListUtils.partition(images, batchSize)
                .stream()
                .map(this::processBatch)
                .collect(Collectors.toList());
    }

    private MnistImageBatch processBatch(List<MnistImage> currentBatch) {
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

    private void showOperations (Graph graph) {
        var iter = graph.operations();
        while (iter.hasNext()) {
            System.out.println(""+iter.next());
        }
    }

    public void updateLayersWeights (Session session, List<LayerWeights> weights) {
        var layerNames = weights.stream()
                .map(LayerWeights::getLayerName)
                .toArray(String[]::new);
        var resultTensors = getTensorsForLayers(session, layerNames);
        IntStream.range(0, weights.size()).forEach( i -> {
            var weightsBuffer = weights.get(i).getWeights();
            weightsBuffer.clear();
            resultTensors.get(i).writeTo(weightsBuffer);
        });
    }

    private List<LayerWeights> prepareLayerWeights(Session session, String[] layerNames) {
        var resultTensors = getTensorsForLayers(session, layerNames);
        final var i = new AtomicInteger();
        return resultTensors.stream()
                .map(tensor -> layerWeightsForTensor (layerNames[i.getAndIncrement()], tensor))
                .collect(Collectors.toList());

    }

    private List<Tensor<?>> getTensorsForLayers(Session session, String[] layerNames) {
        var runner = session.runner();
        for (var layerName : layerNames) {
            runner = runner.fetch(layerName);
        }
        return runner.run();
    }

    private LayerWeights layerWeightsForTensor(String name, Tensor<?> tensor) {
        long[] size = tensor.shape();
        var totalSize = Arrays.stream(size).reduce(Math::multiplyExact);
        return new LayerWeights(name, (int)totalSize.getAsLong());
    }

    @Override
    public void main() throws Throwable {
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

            String[] layerNames = {"hidden1/weights", "hidden1/biases", "hidden2/weights", "hidden2/biases"};
            layersWithWeights = prepareLayerWeights (sess, layerNames);
            var start = System.nanoTime();
            testImagesBatches.forEach( batch -> {
                sess.runner().feed("X", batch.getPixels())
                        .feed("y", batch.getLabel())
                        .addTarget("train/optimize")
                        .run();
                updateLayersWeights(sess, layersWithWeights);
                List<LayerWeights> reduced = performCommunication (layersWithWeights);
            });
            var stop = System.nanoTime();
            System.out.println("Time = "  + (stop - start)*1e-9);
        }
    }

    @Storage(MnistGradientDescent.class)
    enum Shared {
        layersWithWeights;
    }
    private List<LayerWeights> layersWithWeights;

    private List<LayerWeights> performCommunication(List<LayerWeights> layersWithWeights) {
        //interthread-summation

        if (PCJ.myId() == 0) {
            for (var layer : layersWithWeights) {
                layer.getWeights().
            }
        }
    }

    public static void main (String[] args) throws IOException {
        PCJ.start(MnistGradientDescent.class, new NodesDescription("../nodes.txt"));
    }




}
