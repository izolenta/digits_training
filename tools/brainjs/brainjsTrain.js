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

const net = new brain.NeuralNetwork(config);

net.train(
  trainData,
  {
    log: (stats) => console.log(stats),
    errorThresh: 0.005,
    logPeriod: 1,
    iterations: 8,
  }
);

fs.writeFileSync('brainjsTrainingData.json', JSON.stringify(net.toJSON()), {encoding: 'utf8'})

console.log('done');
