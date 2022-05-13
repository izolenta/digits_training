// origin: https://codelabs.developers.google.com/codelabs/tfjs-training-classfication

import tf, {loadLayersModel} from '@tensorflow/tfjs-node'
import loadData from "../data/loader.js";
import fs from "fs";

const model = await loadLayersModel('file://tensorflowTrainingData/model.json');

const {dataArray, labelsArray} = loadData();

const data = tf.tensor2d(dataArray[195], [1, 784]);
const input = data.reshape([1, 28, 28, 1]);

model.predict(input).print();
console.log(labelsArray[195]);

input.dispose();

console.log('done');