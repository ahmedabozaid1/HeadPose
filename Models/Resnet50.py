from keras.layers import Dense, Flatten
from keras.models import Model
from keras.applications.resnet import ResNet50
from keras.preprocessing.image import ImageDataGenerator
from glob import glob
import matplotlib.pyplot as plt
from keras import backend as K
# re-size all the images to this
IMAGE_SIZE = [64, 64]
train_path = '../Data/Train'
test_path = '../Data/Test'



# useful for getting number of classes
folders = glob('../Data/Test/*')
# add preprocessing layer to the front 
resnet = ResNet50(input_shape=IMAGE_SIZE + [3],
            weights='imagenet', include_top=False)

# don't train existing weights
for layer in resnet.layers:
  layer.trainable = False

  # useful for getting number of classes


# our layers - you can add more if you want
x = Flatten()(resnet.output)
# x = Dense(1000, activation='relu')(x)
prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=resnet.input, outputs=prediction)

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
    epochs=200,
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
model.save('resnet50.h5')
