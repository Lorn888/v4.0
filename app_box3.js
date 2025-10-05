let video, model, embeddings, boxes;

async function initCamera() {
  video = document.getElementById('video');
  const stream = await navigator.mediaDevices.getUserMedia({video:{facingMode:'environment'}});
  video.srcObject = stream;
  await new Promise(r => video.onloadedmetadata = r);
}

function cosineSim(a,b){
  let dot=0, na=0, nb=0;
  for(let i=0;i<a.length;i++){dot+=a[i]*b[i]; na+=a[i]*a[i]; nb+=b[i]*b[i];}
  return dot/(Math.sqrt(na)*Math.sqrt(nb)+1e-10);
}

// Capture multiple frames and average embeddings
async function captureMultipleFrames(n=3, delay=100){
  const embeddingsFrames=[];
  for(let i=0;i<n;i++){
    const canvas = document.createElement('canvas');
    canvas.width = 224; canvas.height = 224;
    canvas.getContext('2d').drawImage(video,0,0,224,224);

    const img = tf.browser.fromPixels(canvas).toFloat();
    const normalized = img.sub(tf.scalar(127.5)).div(tf.scalar(127.5)).expandDims(0);
    const emb = model.infer(normalized,true);
    embeddingsFrames.push(Array.from(emb.dataSync()));
    emb.dispose(); img.dispose(); normalized.dispose();

    await new Promise(r => setTimeout(r, delay));
  }
  // Average embeddings across frames
  return embeddingsFrames[0].map((_,i)=>embeddingsFrames.reduce((sum,e)=>sum+e[i],0)/embeddingsFrames.length);
}

function findPoster(vec){
  let best=null, bestScore=-1;
  for(const id of Object.keys(embeddings)){
    for(const emb of embeddings[id]){ // all 3 embeddings per poster
      const score = cosineSim(vec, emb);
      if(score > bestScore){bestScore=score; best=id;}
    }
  }
  return {id:best, score:bestScore};
}

function findBox(id){
  const num=parseInt(id,10);
  for(const b of boxes) if(num>=b.from && num<=b.to) return b.box;
  return null;
}

async function captureAndIdentify(){
  const vec = await captureMultipleFrames(3,100);
  const {id, score} = findPoster(vec);
  const box = findBox(id);

  document.getElementById('result').innerHTML =
    score >= 0.7 ?
    `Poster: ${id}<br>Box: ${box}<br>Confidence: ${score.toFixed(3)}` :
    `No confident match (top: ${id ?? 'n/a'}, score: ${score.toFixed(3)})`;
}

window.addEventListener('load', async ()=>{
  await initCamera();
  document.getElementById('identifyBtn').disabled = true;
  document.getElementById('result').innerText = 'Loading model and embeddings...';
  embeddings = await (await fetch('embeddings_box3.json')).json();
  boxes = await (await fetch('boxes_box3.json')).json();
  // Convert embeddings from array-of-arrays JSON into usable format
  for(const id of Object.keys(embeddings)){
    // ensure embeddings[id] is array of arrays
    if(!Array.isArray(embeddings[id][0])) embeddings[id] = [embeddings[id]];
  }
  model = await mobilenet.load({version:2, alpha:1.0});
  document.getElementById('result').innerText = 'Ready';
  document.getElementById('identifyBtn').disabled = false;
  document.getElementById('identifyBtn').addEventListener('click', captureAndIdentify);
});