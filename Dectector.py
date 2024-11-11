import tenserflow as tf
(x_train, y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()
x_train , x_test = x_train/255.0, x_test/255.0
input_layer = tf.keras.layers.Flatten(input_shape=(28, 28))
hidden1 = tf.keras.layers.Dense(16,activation='sigmoid')
hidden2 = tf.keras.layers.Dense(16,activation = 'sigmoid')
out_put_layer = tf.keras.layers.Dense(10,activation='softmax')


models = tf.keras.Sequential(
    [input_layer,
     hidden1,
     hidden2,
     out_put_layer

     ]
)
models.compile(optimizer='sgd',
loss = 'sparse_categorical_crossentropy',
metrics = ['accuracy']
)

history = models.fit(x_train,y_train,epochs=5)
print(history.history['accuracy'])


