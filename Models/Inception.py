from keras.layers import Dense
from keras.layers.pooling import GlobalAveragePooling2D
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from glob import glob
import matplotlib.pyplot as plt
import tensorflow as tf
# re-size all the images to this
IMAGE_SIZE = [299, 299]

train_path = '../Data/Train'
test_path = '../Data/Test'
  # useful for getting number of classes
folders = glob('../Data/Test/*')

# add preprocessing layer to the front
inception = tf.keras.applications.InceptionV3(
    weights='imagenet', include_top=False, input_shape=(299, 299, 3))
# don't train existing weights
for layer in inception.layers:
  layer.trainable = False

# our layers - you can add more if you want
x = inception.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
# x = Dense(1000, activation='relu')(x)
prediction = Dense(len(folders), activation='softmax')(x)
# create a model object
model = Model(inputs=inception.input, outputs=prediction)
# view the structure of the model
model.summary()


# tell the model what cost and optimization method to use
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
batchSize = 32
train_datagen = ImageDataGenerator(rescale=1./255,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=False)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(train_path,
                                                 target_size=IMAGE_SIZE,
                                                 batch_size=batchSize,
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory(test_path,
                                            target_size=IMAGE_SIZE,
                                            batch_size=batchSize,
                                            class_mode='categorical')


# fit the model
r = model.fit(
    training_set,
    validation_data=test_set,
    epochs=25,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set)
)

loss, accuracy = model.evaluate(training_set, verbose=0)

print('loss')
print(loss)
print('accuracy')
print(accuracy)

# loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# accuracies
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')


print("done")
model.save('inception.h5')
