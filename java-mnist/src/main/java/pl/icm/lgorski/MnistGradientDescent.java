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
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

//cf. https://github.com/tensorflow/models/blob/master/samples/languages/java/training/src/main/java/Train.java

@RegisterStorage(MnistGradientDescent.Shared.class)
public class MnistGradientDescent implements StartPoint {

    private static final int BATCH_SIZE = 100;
    private static final int IMAGE_SIZE = 28*28;
    private static final int COMMUNICATE_AFTER_N_BATCHES = 20;


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

    private static class WeightsTensor {
        @Getter
        private final FloatBuffer weights;

        @Getter
        private final String layerName;

       @Getter
       private final long[] shape;

        public WeightsTensor(String layerName, int size, long[] shape){
            weights = FloatBuffer.allocate(size);
            this.layerName = layerName;
            this.shape = shape;
        }
    }

    private List<MnistImage> readMnistImages (List<String> lines) {
        return lines.stream()
                .map(line -> line.split(" "))
                .map( numbers -> {
                    int label = Integer.parseInt(numbers[0]);
                    Float[] pixels = Arrays.stream(numbers).skip(1).map(Float::valueOf).toArray(Float[]::new);
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

        Tensor<Integer> tensorYs = Tensor.create(new long[] { numberOfElements }, ys);
        Tensor<Float> tensorXs = Tensor.create(new long[] { numberOfElements, IMAGE_SIZE}, xs);
        return new MnistImageBatch(tensorYs, tensorXs);
    }

    private void showOperations (Graph graph) {
        Iterator<?> iter = graph.operations();
        while (iter.hasNext()) {
            System.out.println(""+iter.next());
        }
    }

    public void saveTensorFlowWeightsToJavaObject(Session session, String weightsName) {
  /*      String[] layerNames = weights.stream()
                .map(WeightsTensor::getLayerName)
                .toArray(String[]::new);
        List<Tensor<?>> resultTensors = getTensorsForWeights(session, layerNames);
        IntStream.range(0, weights.size()).forEach( i -> {
            FloatBuffer weightsBuffer = weights.get(i).getWeights();
            weightsBuffer.clear();
            resultTensors.get(i).writeTo(weightsBuffer);
        });*/
    }

    private List<WeightsTensor> prepareWeights(Session session, String weightNames, String weightLengthName, MnistImageBatch primingBatch) {
        List<Tensor<?>> resultTensors = getTensorsForWeights(session, weightNames, weightLengthName, primingBatch);
        final AtomicInteger i = new AtomicInteger();
        return resultTensors.stream()
                .map(tensor -> weightsForTensor(weightNames + ":" + i.getAndIncrement(), tensor))
                .collect(Collectors.toList());

    }

    private List<Tensor<?>> getTensorsForWeights(Session session, String weightNames, String weightLengthName, MnistImageBatch primingBatch) {
        List<Tensor<?>> lengthOfComputationGradients =
                session.runner()
                        .feed("X", primingBatch.getPixels())
                        .feed("y", primingBatch.getLabel())
                        .fetch(weightLengthName)
                        .run();
        int numberOfWeights = lengthOfComputationGradients.get(0).intValue();

        Session.Runner runner = session.runner();
        runner.feed("X", primingBatch.getPixels())
                .feed("y", primingBatch.getLabel());
        for (int i = 0; i < numberOfWeights; i++) {
            runner.fetch(weightNames, i);
        }
        return runner.run();
    }

    private WeightsTensor weightsForTensor(String name, Tensor<?> tensor) {
        long[] size = tensor.shape();
        OptionalLong totalSize = Arrays.stream(size).reduce(Math::multiplyExact);
        return new WeightsTensor(name, (int)totalSize.getAsLong(), size);
    }

    @Override
    public void main() throws Throwable {
        myId = PCJ.myId();
        threadCount = PCJ.threadCount();

        final byte[] graphDef = Files.readAllBytes(Paths.get("../graph.pb"));
        final List<MnistImage> trainImages = readMnistImages(Files.readAllLines(Paths.get("../mnist.train.txt")));
        final List<MnistImage> testImages = readMnistImages(Files.readAllLines(Paths.get("../mnist.test.txt")));

        List<MnistImage> trainImagesSlice = trainImages.subList(BATCH_SIZE * myId, BATCH_SIZE * (myId + 1));
        List<MnistImageBatch> trainImagesBatches = batchMnist(trainImages, BATCH_SIZE);
        final List<MnistImageBatch> testImagesBatches = batchMnist(testImages, 1);

        try (Graph graph = new Graph();
             Session sess = new Session((graph))) {
            graph.importGraphDef(graphDef);
            sess.runner().addTarget("init").run();

            weights = prepareWeights(sess,
                    "train/compute_gradients",
                    "train/compute_gradients_output_length",
                    trainImagesBatches.get(0));

            long start = System.nanoTime();
            AtomicInteger batchCounter = new AtomicInteger();
            testImagesBatches.forEach( batch -> {
                Session.Runner runner = sess.runner().feed("X", batch.getPixels())
                        .feed("y", batch.getLabel());

                if (batchCounter.getAndIncrement() != COMMUNICATE_AFTER_N_BATCHES) {
                    //    saveTensorFlowWeightsToJavaObject(weightsTensor);
                    performCommunication();
                    saveJavaObjectWeightsToTensorFlow(sess, weights);
                    batchCounter.set(0);
                    if (myId == 0) {
                        System.out.println("Communicated");
                    }
                } else {
                    //TODO: podzielić na ścieżki wykonania - z pobieraniem tensorów i bez tego
                    List<Tensor<?>> weightsTensor = sess.runner().feed("X", batch.getPixels())
                            .feed("y", batch.getLabel())
                            .fetch("train/compute_gradients:0")
                            .fetch("train/compute_gradients:1")
                            .fetch("train/compute_gradients:2")
                            .fetch("train/compute_gradients:3")
                            .fetch("train/compute_gradients:4")
                            .fetch("train/compute_gradients:5")
                            .run();
                }
            });
            long stop = System.nanoTime();
            System.out.println("Time = "  + (stop - start)*1e-9 + "Communication time = " + commTime);
        }
    }

    private void saveJavaObjectWeightsToTensorFlow(Session sess, List<WeightsTensor> layersWithWeights) {
        Session.Runner runner = sess.runner();
        for (int i = 0; i < layersWithWeights.size(); i++) {
            WeightsTensor weightsTensor = layersWithWeights.get(i);
            weightsTensor.getWeights().flip();
            Tensor<Float> tensor = Tensor.create(weightsTensor.getShape(), weightsTensor.getWeights());
            runner = runner.feed(weightsTensor.getLayerName(), tensor);
        }
        runner.addTarget("dnn/no_op").run();
    }

    @Storage(MnistGradientDescent.class)
    enum Shared {
        layersWeightsCommunicated
    }
    private List<WeightsTensor> weights;

    private List<float[]> layersWeightsCommunicated;
    int threadCount;
    int myId;

    float commTime = 0;
    private void performCommunication() {
        long start = System.nanoTime();
        PCJ.barrier();
        layersWeightsCommunicated = weights.stream().map(WeightsTensor::getWeights).map(FloatBuffer::array).collect(Collectors.toList());
        PCJ.barrier();
        //allToAllHypercube (); //cf. e.g. http://parallelcomp.uw.hu/ch04lev1sec3.html
        allToAllSimple();
        divideByThreadCount();
        long stop = System.nanoTime();
        commTime += (stop - start) * 1e-9;
        PCJ.barrier();
    }

    private void allToAllSimple () {
        if (myId == 0) {
            PcjFuture<List<float[]>>[] futures = new PcjFuture[threadCount];
            for (int i = 0; i < futures.length; i++) {
                if (i != myId) {
                    futures[i] = PCJ.asyncGet(i, Shared.layersWeightsCommunicated);
                }
            }
            int downloaded = 0;
            while (downloaded != threadCount - 1) {
                for (int i = 0; i < futures.length; i++) {
                    if (futures[i] != null && futures[i].isDone()) {
                        List<float[]> remoteWeights = futures[i].get();
                        addCommunicatedWeightsToLayers(remoteWeights);
                        futures[i] = null;
                        downloaded++;
                    }
                }
            }
            PCJ.broadcast(weights, Shared.layersWeightsCommunicated);
        } else {
            PCJ.waitFor(Shared.layersWeightsCommunicated);
        }
        for (int i = 0; i < weights.size(); i++) {
            FloatBuffer weights = this.weights.get(i).getWeights();
            weights.clear();
            weights.put(layersWeightsCommunicated.get(i));
        }
        PCJ.barrier();
    }
    private void allToAllHypercube() {
        final int d = Integer.numberOfTrailingZeros(threadCount);
        for (int i = 0; i < d; i++) {
            int partner = myId ^ (1 << i);

            List<float[]> rawWeights = weights.stream()
                    .map(WeightsTensor::getWeights)
                    .map(FloatBuffer::array)
                    .collect(Collectors.toList());


            PCJ.asyncPut(rawWeights, partner, Shared.layersWeightsCommunicated);
            PCJ.waitFor(Shared.layersWeightsCommunicated);

            addCommunicatedWeightsToLayers(layersWeightsCommunicated);
            PCJ.barrier();
        }
    }

    private void addCommunicatedWeightsToLayers(List<float[]> communicated) {
        for (int layer = 0; layer < weights.size(); layer++) {
            float[] weightsArray = weights.get(layer).getWeights().array();
            float[] communicatedWeightsArray = communicated.get(layer);
            for (int i = 0; i < weightsArray.length; i++) {
                weightsArray[i] += communicatedWeightsArray[i];
            }
        }
    }


    private void divideByThreadCount() {
        for (WeightsTensor layer : weights) {
            float[] weightsArray = layer.getWeights().array();
            for (int i = 0; i < weightsArray.length; i++) {
                weightsArray[i] /= threadCount;
            }
        }
    }

    public static void main (String[] args) throws IOException {
        PCJ.start(MnistGradientDescent.class, new NodesDescription("../nodes.txt"));
    }




}
