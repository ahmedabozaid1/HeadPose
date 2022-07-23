
from keras.layers import  Dense, Flatten
from keras.models import Model
from keras.applications.vgg19 import VGG19
from keras.preprocessing.image import ImageDataGenerator
from glob import glob
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.utils.vis_utils import plot_model

# re-size all the images to this
IMAGE_SIZE = [64, 64]

train_path = '../Data/Train'
test_path = '../Data/Test'


# useful for getting number of classes
folders = glob('../Data/Test/*')

# add preprocessing layer to the front 
vgg = VGG19(input_shape=IMAGE_SIZE + [3],
            weights='imagenet', include_top=False)

# don't train existing weights
for layer in vgg.layers:
  layer.trainable = False





# our layers - you can add more if you want
x = Flatten()(vgg.output)
# x = Dense(1000, activation='relu')(x)
prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=vgg.input, outputs=prediction)

# view the structure of the model
model.summary()

'''
epochs = 25
learningRate = 0.1
decayRate = learningRate/epochs
momentum = 0.8


lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=learningRate,
    decay_steps=10000,
    decay_rate=decayRate)

optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
'''

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

keras.callbacks.TensorBoard(
    log_dir="logs",
    histogram_freq=2,
    write_graph=True,
    write_images=False,
    write_steps_per_second=False,
    update_freq="epoch",
    profile_batch=0,
    embeddings_freq=0,
    embeddings_metadata=None,
    
)
tensorboard_callback = keras.callbacks.TensorBoard(log_dir="./log")

# fit the model
r = model.fit(
  training_set,
  validation_data=test_set,
  epochs=10,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set),
  callbacks=[tensorboard_callback]
)
loss, accuracy = model.evaluate(training_set, verbose=0)
plot_model(model, to_file='model_plot.png',
           show_shapes=True, show_layer_names=True)
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
model.save('vgg19.h5')
