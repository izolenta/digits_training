// Run this with Node.js to train the neural network
// For the data format please refer to http://yann.lecun.com/exdb/mnist/

import fs from "fs";
import * as brain from "brain.js";

const labelsArray = [];
const dataArray = [];
const trainData = [];

// reading labels

let offset = 4; //skipping the magic number

const labelsFile = fs.readFileSync('./data/train-labels-idx1-ubyte');
const length = labelsFile.readInt32BE(offset);

offset+=4;

for (let i=0; i<length; i++) {
  const output = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
  const label = labelsFile.readInt8(offset);
  output[label] = 1;
  labelsArray.push(output);
  offset++;
}

// reading images

offset = 16; //skipping magic number and data length, assuming it is the same as in labels file

const dataFile = fs.readFileSync('./data/train-images-idx3-ubyte');
const width = dataFile.readInt32BE(8);
const height = dataFile.readInt32BE(12);

console.log(width, height);

for (let i=0; i<length; i++) {
  let nextData = [];
  for (let j=0; j<width*height; j++) {
    nextData.push(dataFile.readUInt8(offset) < 200? 0 : 1); // 50% brightness filtering
    offset++;
  }
  dataArray.push(nextData);
}

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

fs.writeFileSync('digits_training.json', JSON.stringify(net.toJSON()), {encoding: 'utf8'})

console.log('done');
