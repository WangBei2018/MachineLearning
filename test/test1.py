import tensorflow as tf


(xs, ys),_ = tf.keras.datasets.mnist.load_data()
print('datasets:', xs.shape, ys.shape, xs.min(), xs.max())


xs = tf.convert_to_tensor(xs, dtype=tf.float32) / 255.
db = tf.data.Dataset.from_tensor_slices((xs,ys))
db = db.batch(32).repeat(10)


network = tf.keras.Sequential([tf.keras.layers.Dense(256, activation='relu'),
                     tf.keras.layers.Dense(256, activation='relu'),
                     tf.keras.layers.Dense(256, activation='relu'),
                     tf.keras.layers.Dense(10)])
network.build(input_shape=(None, 28*28))
network.summary()

optimizer = tf.keras.optimizers.SGD(lr=0.01)
acc_meter = tf.keras.metrics.Accuracy()

for step, (x, y) in enumerate(db):

    with tf.GradientTape() as tape:
        # [b, 28, 28] => [b, 784]
        x = tf.reshape(x, (-1, 28*28))
        # [b, 784] => [b, 10]
        out = network(x)
        # [b] => [b, 10]
        y_onehot = tf.one_hot(y, depth=10)
        # print(len(y_onehot))
        # [b, 10]
        loss = tf.square(out-y_onehot)
        # [b]
        loss = tf.reduce_sum(loss) / 32
        # loss = tf.keras.backend.categorical_crossentropy(y_onehot, out)


    acc_meter.update_state(tf.argmax(out, axis=1), y)

    grads = tape.gradient(loss, network.trainable_variables)
    optimizer.apply_gradients(zip(grads, network.trainable_variables))


    if step % 500==0:

        print(step, 'loss:', loss, 'acc:', acc_meter.result().numpy())
        print(step, 'loss:', float(loss), 'acc:', acc_meter.result().numpy())
        acc_meter.reset_states()