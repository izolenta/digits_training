import fs from "fs";

function loadData() {
  const labelsArray = [];
  const dataArray = [];

// reading labels

  let offset = 4; //skipping the magic number

  const labelsFile = fs.readFileSync('../data/train-labels-idx1-ubyte');
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

  const dataFile = fs.readFileSync('../data/train-images-idx3-ubyte');
  const width = dataFile.readInt32BE(8);
  const height = dataFile.readInt32BE(12);

  console.log(width, height);

  for (let i=0; i<length; i++) {
    let nextData = [];
    for (let j=0; j<width*height; j++) {
      nextData.push(dataFile.readUInt8(offset) / 256); // 50% brightness filtering
      offset++;
    }
    dataArray.push(nextData);
  }
  return {length, labelsArray, dataArray};
}

export default loadData;