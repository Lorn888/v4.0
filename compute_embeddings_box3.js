const tf = require('@tensorflow/tfjs');
const mobilenet = require('@tensorflow-models/mobilenet');
const fs = require('fs');
const path = require('path');
const { createCanvas, loadImage } = require('canvas');

async function loadImageTensor(filePath) {
  const img = await loadImage(filePath);
  const canvas = createCanvas(224, 224);
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0, 224, 224);
  const tensor = tf.browser.fromPixels(canvas).toFloat();
  const normalized = tensor.sub(127.5).div(127.5).expandDims(0);
  return normalized;
}

async function main() {
  const datasetDir = './dataset';
  const posterIds = fs.readdirSync(datasetDir).filter(f => fs.statSync(path.join(datasetDir,f)).isDirectory() && parseInt(f)>=406 && parseInt(f)<=460);
  const model = await mobilenet.load({version: 2, alpha:1.0});
  const embeddings = {};

  for (const posterId of posterIds) {
    const posterPath = path.join(datasetDir, posterId);
    const files = fs.readdirSync(posterPath).filter(fn => /\.(jpg|jpeg|png)$/i.test(fn));
    const vectors = [];
    for (const file of files) {
      const tensor = await loadImageTensor(path.join(posterPath, file));
      const embedding = model.infer(tensor, true);
      vectors.push(Array.from(embedding.dataSync()));
      tensor.dispose(); embedding.dispose();
    }
    // average the 3 images
    const avg = vectors[0].map((_, i) => vectors.reduce((sum,v)=>sum+v[i],0)/vectors.length);
    embeddings[posterId] = avg;
  }

  fs.writeFileSync('embeddings_box3.json', JSON.stringify(embeddings));
  console.log('Saved embeddings_box3.json with', Object.keys(embeddings).length, 'posters');
}

main();