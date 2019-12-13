import { buildConvNetModelForMnist, train } from "./model.mjs";
import { MnistData } from "./data.mjs";

let canvas, rawImage, model, ctx, pos = { x: 0, y: 0 };

function init() {
    canvas = document.getElementById('canvas');
    rawImage = document.getElementById('canvasImg');
    ctx = canvas.getContext('2d');
    ctx.fillRect(0, 0, 280, 280);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mousedown', setPosition);
    canvas.addEventListener('mouseenter', setPosition);
    document.getElementById('sb').addEventListener('click', save);
    document.getElementById('cb').addEventListener('click', erase);
}

function setPosition(e) {
    pos.x = e.clientX - 100;
    pos.y = e.clientY - 100;
}

function save() {
    var raw = tf.browser.fromPixels(rawImage, 1);
    var resized = tf.image.resizeBilinear(raw, [28, 28]);
    var tensor = resized.expandDims(0);
    var prediction = model.predict(tensor);
    var pIndex = tf.argMax(prediction, 1).dataSync();
    alert(pIndex);
}

function draw(e) {
    if (e.buttons != 1) {
        return;
    }

    ctx.beginPath();
    ctx.lineWidth = 24;
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'white';
    ctx.moveTo(pos.x, pos.y);
    setPosition(e);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
    rawImage.src = canvas.toDataURL('image/png');
}

function erase() {
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, 280, 280);
}

async function run() {
    const data = new MnistData();
    await data.load();
    model = buildConvNetModelForMnist();

    tfvis.show.modelSummary({ name: 'Model Architecture' }, model);


    await train(model, data);

    init();
    alert("Training is done, try classifying your handwriting!");


}

document.addEventListener('DOMContentLoaded', run);
