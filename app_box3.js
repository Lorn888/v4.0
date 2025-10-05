let video, model, embeddings, boxes;

async function initCamera() {
  video = document.getElementById('video');
  const stream = await navigator.mediaDevices.getUserMedia({video:{facingMode:'environment'}});
  video.srcObject = stream;
  await new Promise(r=>video.onloadedmetadata=r);
}

function cosineSim(a,b){
  let dot=0, na=0, nb=0;
  for(let i=0;i<a.length;i++){dot+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i];}
  return dot/(Math.sqrt(na)*Math.sqrt(nb)+1e-10);
}

async function loadResources(){
  embeddings = await (await fetch('embeddings_box3.json')).json();
  boxes = await (await fetch('boxes_box3.json')).json();
  model = await mobilenet.load({version:2, alpha:1.0});
}

function findPoster(vec){
  let best=null, bestScore=-1;
  for(const id of Object.keys(embeddings)){
    const score=cosineSim(vec, embeddings[id]);
    if(score>bestScore){bestScore=score;best=id;}
  }
  return {id:best, score:bestScore};
}

function findBox(id){
  const num=parseInt(id,10);
  for(const b of boxes) if(num>=b.from && num<=b.to) return b.box;
  return null;
}

async function captureAndIdentifyMultipleFrames(frames=5){
  let avgVec = null;

  for(let i=0;i<frames;i++){
    const canvas=document.createElement('canvas'); canvas.width=224; canvas.height=224;
    canvas.getContext('2d').drawImage(video,0,0,224,224);
    const img=tf.browser.fromPixels(canvas).toFloat();
    const normalized=img.sub(tf.scalar(127.5)).div(tf.scalar(127.5)).expandDims(0);
    const embedding=model.infer(normalized,true);
    const vec=Array.from(embedding.dataSync());
    embedding.dispose(); img.dispose(); normalized.dispose();

    if(!avgVec) avgVec=vec;
    else avgVec = avgVec.map((v,i)=>v+vec[i]);
    await new Promise(r=>setTimeout(r,100)); // 100ms between frames
  }

  avgVec = avgVec.map(v=>v/frames);
  const {id, score}=findPoster(avgVec);
  const box=findBox(id);
  document.getElementById('result').innerHTML = score>=0.62 ? 
    `Poster: ${id}<br>Box: ${box}<br>Confidence: ${score.toFixed(3)}` :
    `No confident match (top: ${id ?? 'n/a'}, score: ${score.toFixed(3)})`;
}

  const {id, score}=findPoster(vec);
  const box=findBox(id);
  document.getElementById('result').innerHTML = score>=0.62 ? 
    `Poster: ${id}<br>Box: ${box}<br>Confidence: ${score.toFixed(3)}` :
    `No confident match (top: ${id ?? 'n/a'}, score: ${score.toFixed(3)})`;
}

window.addEventListener('load', async ()=>{
  await initCamera();
  await loadResources();
  document.getElementById('identifyBtn').addEventListener('click', captureAndIdentify);
});