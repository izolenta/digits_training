// Run this with Node.js to train the neural network
// For the data format please refer to http://yann.lecun.com/exdb/mnist/

import fs from "fs";
import * as brain from "brain.js";
import loadData from "../data/loader.js";

const trainData = [];

const {length, dataArray, labelsArray} = loadData();

const config = {
  hiddenLayers: [80, 10],
  activation: 'sigmoid',
};

for (let i=0; i<length; i++) {
  trainData.push({
    'input': dataArray[i], 'output': labelsArray[i]
  })
}

const trainingData = fs.readFileSync('./digits_training.json');
const net = new brain.NeuralNetwork(config)

net.fromJSON(JSON.parse(trainingData.toString()));

for (let i=0; i<28; i++) {
  let line = '';
  for (let j=0; j<28; j++) {
    line+=(dataArray[125][i*28+j] === 0? ' ' : 'X');
  }
  console.log(line);
}