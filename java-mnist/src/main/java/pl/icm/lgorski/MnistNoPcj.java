package pl.icm.lgorski;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import org.apache.commons.collections4.ListUtils;
import org.apache.commons.lang3.ArrayUtils;
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
import java.sql.SQLOutput;
import java.util.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

//cf. https://github.com/tensorflow/models/blob/master/samples/languages/java/training/src/main/java/Train.java

public class MnistNoPcj {

    private static final int BATCH_SIZE = 50;
    private static final int IMAGE_SIZE = 28*28;
    private static final int EPOCHS = 20;
    public static final float LEARNING_RATE = 0.01f;


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

    public static void main(String[] args) throws Throwable {
        MnistNoPcj mnist = new MnistNoPcj();
        mnist.main();
    }
    public void main() throws Throwable {

        final byte[] graphDef = Files.readAllBytes(Paths.get("../graph.pb"));

        // MNIST's main dataset contains of 60 000 images. Of that we are using 55 000 for training
        // and using 5 000 as validation set. Cf. A. Géron, "Hands-On Machine Learning with
        // SciKit-Learn and Tensorflow",
        List<MnistImage> trainImages = readMnistImages(Files.readAllLines(Paths.get("../mnist.train.txt")))
                .stream().skip(5_000).collect(Collectors.toList());
        final List<MnistImage> testImages = readMnistImages(Files.readAllLines(Paths.get("../mnist.train.txt")))
                .stream().limit(5_000).collect(Collectors.toList());

        //  addGaussianNoiseToImages(trainImages);

        List<MnistImageBatch> trainImagesBatches = batchMnist(trainImages, BATCH_SIZE);
        final List<MnistImageBatch> testImagesBatches = batchMnist(testImages, testImages.size());

        final float learningRate = LEARNING_RATE;
        final Tensor<Float> learningRateTensor = createScalarTensor(learningRate);

        ConfigProto config = ConfigProto.newBuilder().setIntraOpParallelismThreads(24).setInterOpParallelismThreads(2).setAllowSoftPlacement(true).build();
       // ConfigProto config = ConfigProto.getDefaultInstance();
        System.out.println("intra = " + config.getIntraOpParallelismThreads() + " inter = " + config.getInterOpParallelismThreads());
        try (Graph graph = new Graph();
             Session sess = new Session(graph, config.toByteArray())) {

            graph.importGraphDef(graphDef);
            for (String runType : new String[] { "warmUp", "mainRun"}) {
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
                long start = System.nanoTime();
                for (int epoch = 0; epoch < EPOCHS; epoch++) {
                    Collections.shuffle(trainImages);
                    List<MnistImageBatch> batches = batchMnist(trainImages, BATCH_SIZE);
                    final int finalEpoch = epoch;
                    batches.forEach(batch -> {
                        imagesProcessed.addAndGet(batch.label.numElements());
                        batchesProcessed.incrementAndGet();

                            Session.Runner runner2 = sess.runner();
                            runner2 = runner2.feed("X", batch.getPixels())
                                    .feed("y", batch.getLabel())
                                    .feed("learning_rate", learningRateTensor);
                            List<Tensor<?>> weightsTensor = runner2
                                    .addTarget("train_simple/optimize")
                                    .run();
                            closeListOfTensors(weightsTensor);
                        batch.getPixels().close();
                    });
                    float accuracy = evaluate(sess, testImagesBatches.get(0));
                    System.out.println(accuracy);
                }
                long stop = System.nanoTime();
                float accuracy = evaluate(sess, testImagesBatches.get(0));
                    System.out.println("Time(" + runType + ") = " + (stop - start) * 1e-9 +
                            " accuracy = " + accuracy +
                            " images processed (per thread) = " + imagesProcessed.get() +
                            " batches processed (per thread) = " + batchesProcessed.get());
            }
        }
    }

    private Tensor<Float> createScalarTensor(float value) {
        final FloatBuffer buffer = FloatBuffer.allocate(1);
        ((Buffer)buffer.put(value)).flip();
        return Tensor.create(new long[]{}, buffer);
    }



    private List<MnistImage> divideImagesAmongThreads(List<MnistImage> trainImages, Predicate<Integer> divisionPolicy) {
        final List<MnistImage> tmp = trainImages;
        trainImages = IntStream.range(0, trainImages.size())
                .filter(divisionPolicy::test)
                .mapToObj(tmp::get)
                .collect(Collectors.toList());
        return trainImages;
    }

    java.util.Random r = new java.util.Random();

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


    private List<LayerTensor> weights;

}
