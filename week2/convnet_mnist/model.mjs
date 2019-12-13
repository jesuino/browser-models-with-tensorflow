export function buildConvNetModelForMnist() {
    const model = tf.sequential();

    model.add(tf.layers.conv2d({
        inputShape: [28, 28, 1],
        kernelSize: 3,
        filters: 8,
        activation: 'relu'
    }));

    model.add(tf.layers.maxPooling2d({
        poolSize: [2, 2]
    }));

    model.add(tf.layers.conv2d({
        filters: 16,
        kernelSize: 3,
        activation: 'relu'
    }));

    model.add(tf.layers.maxPooling2d({
        poolSize: [2, 2]
    }));

    model.add(tf.layers.conv2d({
        filters: 16,
        kernelSize: 3,
        activation: 'relu'
    }));

    model.add(tf.layers.maxPooling2d({
        poolSize: [2, 2]
    }));

    model.add(tf.layers.flatten());

    model.add(tf.layers.dense({
        units: 128,
        activation: 'relu'
    }));

    model.add(tf.layers.dense({
        units: 10,
        activation: 'softmax'
    }));

    model.compile({
        optimizer: tf.train.adam(),
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    return model;
}

export async function train(model, data) {
    const BATCH_SIZE = 512;
    const TRAIN_DATA_SIZE = 5500;
    const TEST_DATA_SIZE = 1000;

    const fitCallbacks = tfvis.show.fitCallbacks({
        name: 'Model Training',
        styles: { height: '1000px' }
    }, ['loss', 'val_loss', 'acc', 'val_acc']);

    const [trainXs, trainYs] = tf.tidy(() => {
        const d = data.nextTrainBatch(TRAIN_DATA_SIZE);
        return [
            d.xs.reshape([TRAIN_DATA_SIZE, 28, 28, 1]),
            d.labels
        ];
    });

    const [testXs, testYs] = tf.tidy(() => {
        const d = data.nextTestBatch(TEST_DATA_SIZE);
        return [
            d.xs.reshape([TEST_DATA_SIZE, 28, 28, 1]),
            d.labels
        ]
    });

    return model.fit(trainXs, trainYs, {
        batchSize: BATCH_SIZE,
        validationData: [testXs, testYs],
        epochs: 20,
        shuffle: true,
        callbacks: fitCallbacks
    });

}