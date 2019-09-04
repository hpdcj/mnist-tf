package pl.icm.lgorski;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.apache.commons.collections4.ListUtils;
import org.apache.commons.lang3.ArrayUtils;
import org.pcj.*;
import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.framework.ConfigProto;

import java.io.IOException;
import java.nio.Buffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

//cf. https://github.com/tensorflow/models/blob/master/samples/languages/java/training/src/main/java/Train.java

@RegisterStorage(MnistGradientDescentAsync.Shared.class)
public class MnistGradientDescentAsync implements StartPoint {

    private static final int BATCH_SIZE = 50;
    private static final int IMAGE_SIZE = 28*28;
    private static final int COMMUNICATE_AFTER_N_EPOCHS = 1;
    private static final int EPOCHS = 20;
    public static final float LEARNING_RATE = 0.01f;
    public static final int NUM_NEIGHBORS = 1;

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

    private static class LayerTensor {
        @Getter
        private final FloatBuffer weights;

        @Getter
        private final String layerName;

       @Getter
       private final long[] shape;

        public LayerTensor(String layerName, long[] shape){
            long totalSize = Arrays.stream(shape).reduce(Math::multiplyExact).getAsLong();
            weights = FloatBuffer.allocate((int)totalSize);
            this.layerName = layerName;
            this.shape = shape;
        }
    }

    private List<MnistImage> readMnistImages (List<String> lines) {
        return lines.stream()
                .map(line -> line.split(" "))
                .map( numbers -> {
                    int label = Integer.parseInt(numbers[0]);
                    Float[] pixels = Arrays.stream(numbers)
                            .skip(1)
                            .map(Float::valueOf)
                            .toArray(Float[]::new);
                    return new MnistImage(label, ArrayUtils.toPrimitive(pixels));
                })
                .collect(Collectors.toList());
    }

    private List<MnistImageBatch> batchMnist (List<MnistImage> images, int batchSize) {
        return ListUtils.partition(images, batchSize)
                .stream()
                .map(this::tensorsFromImages)
                .collect(Collectors.toList());

    }

    private MnistImageBatch tensorsFromImages(List<MnistImage> currentBatch) {
        int numberOfElements = currentBatch.size();
        IntBuffer ys = IntBuffer.allocate(numberOfElements);
        FloatBuffer xs = FloatBuffer.allocate(numberOfElements * IMAGE_SIZE);

        currentBatch.stream()
                .forEach( image -> {
                    ys.put(image.getLabel());
                    xs.put(image.getPixels());
                });

        //casting to Buffer to overcome NoSuchMethodError exception arising for certain Java versions,
        //cf. e.g. https://github.com/plasma-umass/doppio/issues/497
        ((Buffer)ys).flip();
        ((Buffer)xs).flip();

        Tensor<Integer> tensorYs = Tensor.create(new long[] { numberOfElements }, ys);
        Tensor<Float> tensorXs = Tensor.create(new long[] { numberOfElements, IMAGE_SIZE}, xs);
        return new MnistImageBatch(tensorYs, tensorXs);
    }

    public void saveTensorFlowWeightsToJavaObject(List<Tensor<?>> tensorsWeights, List<LayerTensor> javaWeights) {
        IntStream.range(0, javaWeights.size())
                .forEach( i -> {
                    FloatBuffer weightsBuffer = javaWeights.get(i).getWeights();
                    ((Buffer)weightsBuffer).clear();
                    tensorsWeights.get(i).writeTo(weightsBuffer);
                    ((Buffer)weightsBuffer).flip();
                });
    }

    private List<LayerTensor> initialWeights(Session session, String[] weightNames, MnistImageBatch primingBatch) {
        List<Tensor<?>> resultTensors = null;
        try {
            resultTensors = getTensorsForWeights(session, weightNames, primingBatch);
            final AtomicInteger i = new AtomicInteger();
            return resultTensors.stream()
                    .map(tensor -> weightsForTensor(weightNames[i.getAndIncrement()], tensor))
                    .collect(Collectors.toList());
        } finally {
            closeListOfTensors(resultTensors);
        }
    }

    private List<Tensor<?>> getTensorsForWeights(Session session, String[] weightNames, MnistImageBatch primingBatch) {
        Session.Runner runner = session.runner();
        runner.feed("X", primingBatch.getPixels())
                .feed("y", primingBatch.getLabel());
        Arrays.stream(weightNames)
                .forEach(runner::fetch);
        return runner.run();
    }

    private LayerTensor weightsForTensor(String name, Tensor<?> tensor) {
        long[] size = tensor.shape();
        return new LayerTensor(name, size);
    }

    private void closeListOfTensors(List<Tensor<?>> list) {
        list.stream()
                .forEach(Tensor::close);
    }

    @Override
    public void main() throws Throwable {
        myId = PCJ.myId();
        threadCount = PCJ.threadCount();

        final byte[] graphDef = Files.readAllBytes(Paths.get("graph.pb"));

        // MNIST's main dataset contains of 60 000 images. Of that we are using 55 000 for training
        // and using 5 000 as validation set. Cf. A. Géron, "Hands-On Machine Learning with
        // SciKit-Learn and Tensorflow",
        List<MnistImage> trainImages = readMnistImages(Files.readAllLines(Paths.get("mnist.train.txt")))
                .stream().skip(5_000).collect(Collectors.toList());
        final List<MnistImage> testImages = readMnistImages(Files.readAllLines(Paths.get("mnist.train.txt")))
                .stream().limit(5_000).collect(Collectors.toList());

        trainImages = divideImagesAmongThreads(trainImages, roundRobin);
      //  addGaussianNoiseToImages(trainImages);

        List<MnistImageBatch> trainImagesBatches = batchMnist(trainImages, BATCH_SIZE);
        final List<MnistImageBatch> testImagesBatches = batchMnist(testImages, testImages.size());

        final float learningRate = LEARNING_RATE * threadCount;
        final Tensor<Float> learningRateTensor = createScalarTensor(learningRate);
        ConfigProto config = ConfigProto.newBuilder().setIntraOpParallelismThreads(12).setInterOpParallelismThreads(2).setAllowSoftPlacement(true).build();

        try (Graph graph = new Graph();
             Session sess = new Session((graph), config.toByteArray())) {
            graph.importGraphDef(graphDef);
            for (String runType : new String[] { "warmUp", "mainRun"}) {
                commTime = 0;
                layersWeightsCommunicated = null;
                AtomicInteger imagesProcessed = new AtomicInteger();
                AtomicInteger batchesProcessed = new AtomicInteger();
                closeListOfTensors(sess.runner().addTarget("init").run());

                String[] gradientWeightNames = {
                        "train/gradients/dnn/hidden1/MatMul_grad/tuple/control_dependency_1:0",
                        "train/gradients/dnn/hidden1/BiasAdd_grad/tuple/control_dependency_1:0",
                        "train/gradients/dnn/hidden2/MatMul_grad/tuple/control_dependency_1:0",
                        "train/gradients/dnn/hidden2/BiasAdd_grad/tuple/control_dependency_1:0",
                        "train/gradients/dnn/outputs/MatMul_grad/tuple/control_dependency_1:0",
                        "train/gradients/dnn/outputs/BiasAdd_grad/tuple/control_dependency_1:0",
                };
                weights = initialWeights(sess,
                        gradientWeightNames,
                        trainImagesBatches.get(0));

                closeListOfTensors(sess.runner().addTarget("init").run());
                String[] layerNames = {
                        "hidden1/weights",
                        "hidden1/biases",
                        "hidden2/weights",
                        "hidden2/biases",
                        "outputs/weights",
                        "outputs/biases"
                };
                Session.Runner initRunner = sess.runner();
                for (String layerName : layerNames) {
                    initRunner.fetch(layerName);
                }
                List<Tensor<?>> initialWeights = initRunner.run();
                final long[][] shapes = initialWeights.stream()
                        .map(Tensor::shape)
                        .collect(Collectors.toList())
                        .toArray(new long[][]{});
                if (PCJ.myId() == 0) {
                    AtomicInteger i = new AtomicInteger();
                    List<float[]> forCommunication = initialWeights.stream().map(tensor -> {
                        LayerTensor layerTensor = new LayerTensor(layerNames[i.getAndIncrement()], tensor.shape());
                        tensor.writeTo(layerTensor.getWeights());
                        return layerTensor;
                    })
                            .map(LayerTensor::getWeights)
                            .map(FloatBuffer::array).collect(Collectors.toList());
                    PCJ.broadcast(forCommunication, Shared.layersWeightsCommunicated);
                } else {

                    PCJ.waitFor(Shared.layersWeightsCommunicated);
                    modifyTensorFlowWeights(sess, layerNames, shapes, layersWeightsCommunicated);
                }


                List<float[]> weightsCurrent[] = new List[] {layersWeightsCommunicated};


                PCJ.barrier();
                closeListOfTensors(initialWeights);
                long start = System.nanoTime();

                for (int epoch = 0; epoch < EPOCHS; epoch++) {
                    Collections.shuffle(trainImages);
                    List<MnistImageBatch> batches = batchMnist(trainImages, BATCH_SIZE);
                    final int finalEpoch = epoch;
                    batches.forEach(batch -> {
                        imagesProcessed.addAndGet(batch.label.numElements());
                        batchesProcessed.incrementAndGet();
                        if (finalEpoch % COMMUNICATE_AFTER_N_EPOCHS == 0) {
                            Session.Runner runner = sess.runner().feed("X", batch.getPixels())
                                    .feed("y", batch.getLabel())
                                    .feed("learning_rate", learningRateTensor)
                                    .addTarget("train_simple/optimize");
                            for (String layer : weights.stream().map(LayerTensor::getLayerName).collect(Collectors.toList())) {
                                runner = runner.fetch(layer);
                            }
                            List<Tensor<?>> weightsTensor = runner.run();
                            saveTensorFlowWeightsToJavaObject(weightsTensor, weights);
                            closeListOfTensors(weightsTensor);
                            weightsCurrent[0] = performLocalWeightsCalculation (weightsCurrent[0],
                                    layersWeightsCommunicated,
                                    weights.stream().map(LayerTensor::getWeights).map(FloatBuffer::array).collect(Collectors.toList()),
                                    learningRate);
                            performCommunication(weightsCurrent[0]);
                            modifyTensorFlowWeights(sess, layerNames, shapes, weightsCurrent[0]);
                        } else {
                            Session.Runner runner = sess.runner();
                            runner = runner.feed("X", batch.getPixels())
                                    .feed("y", batch.getLabel())
                                    .feed("learning_rate", learningRateTensor);
                            List<Tensor<?>> weightsTensor = runner
                                    .addTarget("train_simple/optimize")
                                    .run();
                            closeListOfTensors(weightsTensor);
                        }
                        batch.getPixels().close();
                    });
                    float accuracy = evaluate(sess, testImagesBatches.get(0));
                    System.out.println(accuracy);
                }
                long stop = System.nanoTime();
                float accuracy = evaluate(sess, testImagesBatches.get(0));
                if (myId == 0) {
                    System.out.println("Time(" + runType + ") = " + (stop - start) * 1e-9 +
                            " Communication time = " + commTime +
                            " accuracy = " + accuracy +
                            " images processed (per thread) = " + imagesProcessed.get() +
                            " batches processed (per thread) = " + batchesProcessed.get());
                }
            }
        }
    }

    private void modifyTensorFlowWeights(Session sess, String[] layerNames, long[][] shapes, List<float[]> newValue) {
        Session.Runner runner = sess.runner();
        List<Tensor> tensors = new ArrayList<>(layerNames.length);
        for (int i = 0; i < layerNames.length; i++) {
            Tensor tensor = Tensor.create(shapes[i], FloatBuffer.wrap(newValue.get(i)));
            tensors.add(tensor);
            runner = runner.feed("modify_weights/p" + layerNames[i], tensor)
                    .addTarget("modify_weights/assign-" + layerNames[i]);
        }
        closeListOfTensors(runner.run());
        for (Tensor tensor : tensors) {
            tensor.close();
        }
    }


    boolean windowFunction (float[] thisThreadWeights, float[] thisThreadGradients, float[] remoteWeights, float learningRate) {
        float norm1 = 0, norm2 = 0;

        for (int i = 0; i < thisThreadWeights.length; i++) {
            norm1 += Math.pow(thisThreadWeights[i] - learningRate * thisThreadGradients[i] - remoteWeights[i], 2);
            norm2 += Math.pow(thisThreadWeights[i] - remoteWeights[i], 2);
        }
        return norm1 < norm2;
    }
    private List<float[]> performLocalWeightsCalculation(List<float[]> weightsOld, List<float[]> layersWeightsCommunicated, List<float[]> gradients, float learningRate) {
        List<float[]> result = new ArrayList<>(weightsOld.size());
        for (int i = 0; i < weightsOld.size(); i++) {
            float[] thisThreadWeights = weightsOld.get(i);
            float[] thisThreadGradients = gradients.get(i);
            float[] remoteWeights = layersWeightsCommunicated.get(i);
            float[] modifiedWeights = new float[thisThreadWeights.length];
            boolean useRemote = remoteWeights != null && windowFunction(thisThreadWeights, thisThreadGradients, remoteWeights, learningRate);
            for (int j = 0; j < modifiedWeights.length; j++) {
                modifiedWeights[j] = thisThreadWeights[j] - learningRate * thisThreadGradients[j];
                if (useRemote) {
                    modifiedWeights[j]  -= learningRate * (thisThreadWeights[j] - (thisThreadWeights[j] + remoteWeights[j])/2);
                }
            }
            result.add(modifiedWeights);
        }
        return result;
    }

    private Tensor<Float> createScalarTensor(float value) {
        final FloatBuffer buffer = FloatBuffer.allocate(1);
        ((Buffer)buffer.put(value)).flip();
        return Tensor.create(new long[]{}, buffer);
    }

    private Predicate<Integer> roundRobin = i -> i % PCJ.threadCount() == PCJ.myId();
    private Predicate<Integer> weakScaling = i -> true;

    private List<MnistImage> divideImagesAmongThreads(List<MnistImage> trainImages) {
        return divideImagesAmongThreads(trainImages, roundRobin);
    }

    private List<MnistImage> divideImagesAmongThreads(List<MnistImage> trainImages, Predicate<Integer> divisionPolicy) {
        final List<MnistImage> tmp = trainImages;
        trainImages = IntStream.range(0, trainImages.size())
                .filter(divisionPolicy::test)
                .mapToObj(tmp::get)
                .collect(Collectors.toList());
        return trainImages;
    }

    Random r = new Random();

    private void addGaussianNoiseToImages(List<MnistImage> trainImages) {
        for (MnistImage image: trainImages) {
            float[] pixels = image.getPixels();
            for (int i = 0; i < pixels.length; i++) {
                double noise = r.nextGaussian() * Math.sqrt(0.01) + 0;
                pixels[i] += noise;
            }
        }
    }


    private float evaluate(Session sess, MnistImageBatch evaluationBatch) {
        List<Tensor<?>> result =  sess.runner().feed("X", evaluationBatch.getPixels())
                        .feed("y", evaluationBatch.getLabel())
                        .fetch("eval/accuracy").run();

        float floatResult = result.get(0).floatValue();
        closeListOfTensors(result);
        return  floatResult;
    }

    @Storage(MnistGradientDescentAsync.class)
    enum Shared {
        layersWeightsCommunicated
    }
    private List<LayerTensor> weights;

    private List<float[]> layersWeightsCommunicated;


    private int threadCount;
    private int myId;


    private float commTime = 0;
    private void performCommunication(List<float[]> weights) {
        long start = System.nanoTime();
        new Random().ints(0, PCJ.threadCount())
                .filter( i -> i != PCJ.myId())
                .distinct()
                .limit(NUM_NEIGHBORS)
                .forEach( thread -> {
                    PCJ.asyncPut(weights, thread, Shared.layersWeightsCommunicated);
                });
        long stop = System.nanoTime();
        commTime += (stop - start) * 1e-9;

    }

    private void allToAllHypercube() {
        final int d = Integer.numberOfTrailingZeros(threadCount);
        for (int i = 0; i < d; i++) {
            int partner = myId ^ (1 << i);

            List<float[]> rawWeights = weights.stream()
                    .map(LayerTensor::getWeights)
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
        for (LayerTensor layer : weights) {
            float[] weightsArray = layer.getWeights().array();
            for (int i = 0; i < weightsArray.length; i++) {
                weightsArray[i] /= threadCount;
            }
        }
    }

    public static void main (String[] args) throws IOException {
        PCJ.start(MnistGradientDescentAsync.class, new NodesDescription("nodes.txt"));
    }




}
