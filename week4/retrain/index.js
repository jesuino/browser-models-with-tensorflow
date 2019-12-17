let mobilenet;
let model;
const MODEL_URL = "https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json";
const webcam = new Webcam(document.getElementById("wc"));
const dataset = new RPSDataset();
let rockSamples = 0, paperSamples = 0, scissorsSamples = 0;
let isPredicting = false;

async function init() {
    await webcam.setup();
    mobilenet = await loadMobilenet();
}

async function loadMobilenet() {
    const mobilenet = await tf.loadLayersModel(MODEL_URL);
    const layer = mobilenet.getLayer("conv_pw_13_relu");
    return tf.model({ inputs: mobilenet.inputs, outputs: layer.output });
}

function doTraining() {
    train();
}

async function train() {
    dataset.ys = null;
    dataset.encodeLabels(3);
    model = tf.sequential({
        layers: [
            tf.layers.flatten({ inputShape: mobilenet.outputs[0].shape.slice(1) }),
            tf.layers.dense({ units: 100, activation: "relu" }),
            tf.layers.dense({ units: 3, activation: "softmax" })
        ]

    })
    const optimizer = tf.train.adam(0.0001);
    model.compile({ optimizer: optimizer, loss: "categoricalCrossentropy" });
    let loss = 0;
    model.fit(dataset.xs, dataset.ys, {
        epochs: 10,
        callbacks: {
            onBatchEnd: async (batch, logs) => {
                loss = logs.loss.toFixed(5);
                console.log("LOSS: " + loss);
            }
        }
    })
}

function handleButton(elem) {
    switch (elem.id) {
        case "0":
            rockSamples++;
            document.getElementById("rockSamples").innerText = "Rock Samples:" + rockSamples;
            break;
        case "1":
            paperSamples++;
            document.getElementById("paperSamples").innerText = "Paper Samples:" + paperSamples;
            break;
        case "2":
            scissorsSamples++;
            document.getElementById("scissorsSamples").innerText = "Scissors Samples:" + scissorsSamples;
            break;
    }
    label = parseInt(elem.id);
    const img = webcam.capture();
    dataset.addExample(mobilenet.predict(img), label);
}

function startPredicting() {
    isPredicting = true;
    predict();
}

function stopPredicting() {
    isPredicting = false;
    predict();
}

async function predict() {
    let predictionText = "?";
    while (isPredicting) {
        const predictedClass = tf.tidy(() => {
            const img = webcam.capture();
            const activation = mobilenet.predict(img);
            const preditions = model.predict(activation);
            return preditions.as1D().argMax();
        });

        const classId = predictedClass.dataSync()[0];
        predictionText = "?";
        switch (classId) {
            case 0:
                predictionText = "I see Rock";
                break;
            case 1:
                predictionText = "I see Paper";
                break;
            case 2:
                predictionText = "I see Scissors";
                break;
        }

        document.getElementById("prediction").innerText = predictionText;
        predictedClass.dispose();
        await tf.nextFrame();
    }
}
init();