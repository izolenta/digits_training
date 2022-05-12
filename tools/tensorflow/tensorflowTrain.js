// origin: https://codelabs.developers.google.com/codelabs/tfjs-training-classfication

import tf from '@tensorflow/tfjs-node'
import loadData from "../data/loader.js";
import fs from "fs";

const model = tf.sequential()

model.add(tf.layers.conv2d({
  inputShape: [28, 28, 1],
  kernelSize: 5,
  filters: 8,
  strides: 1,
  activation: 'relu',
  kernelInitializer: 'varianceScaling'
}));

model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));

model.add(tf.layers.conv2d({
  kernelSize: 5,
  filters: 16,
  strides: 1,
  activation: 'relu',
  kernelInitializer: 'varianceScaling'
}));

model.add(tf.layers.maxPooling2d({poolSize: [2, 2], strides: [2, 2]}));

model.add(tf.layers.flatten());

model.add(tf.layers.dense({
  units: 10,
  kernelInitializer: 'varianceScaling',
  activation: 'softmax'
}));

const optimizer = tf.train.adam();
model.compile({
  optimizer: optimizer,
  loss: 'categoricalCrossentropy',
  metrics: ['accuracy'],
});

const {length, dataArray, labelsArray} = loadData();

const data = tf.tensor2d(dataArray, [length, 784]);
const labels = tf.tensor2d(labelsArray, [length, 10]);

const [trainXs, trainYs] = tf.tidy(() => {
  return [
    data.reshape([length, 28, 28, 1]),
    labels
  ]
});

await model.fit(trainXs, trainYs, {
  batchSize: 500,
  //validationData: [testXs, testYs],
  epochs: 5,
  shuffle: true,
  //callbacks: fitCallbacks
});

const filename = 'tensorflowTrainingData';

if (fs.existsSync(filename)) {
  fs.rmSync(filename, {recursive: true, force: true});
}

await model.save('file://'+filename);

console.log('done');