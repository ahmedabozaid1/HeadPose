from json import load
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model




valid_path = '../cropped' #cropped


test_datagen = ImageDataGenerator(rescale=1./255)


test_images=test_datagen.flow_from_directory(valid_path,
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='categorical')


test_images2 = test_datagen.flow_from_directory(valid_path,
                                            target_size = (299, 299),
                                            batch_size = 32,
                                            class_mode = 'categorical')

vgg16 = load_model('../Models/vgg16.h5') 
#vgg19 = load_model('../Models/vgg19.h5')
#resnet= load_model('../Models/resnet50.h5')
#cnn= load_model('../Models/cnn.h5')

#inception=load_model('../Models/inception.h5')
#xception=load_model('../Models/xception.h5')

print('6 models loaded')

print('evaluating vgg16')
lossvgg16, accvgg16 = vgg16.evaluate(test_images, verbose=2)
print('evaluating vgg19')
#lossvgg19, accvgg19 = vgg19.evaluate(test_images, verbose=2)
print('evaluating resnet')
#lossresnet, accresnet = resnet.evaluate(test_images, verbose=2)
#print('evaluating cnn')
#losscnn, acccnn = cnn.evaluate(test_images, verbose=2)
print('evaluating incep')
#lossincp, accincp = inception.evaluate(test_images2, verbose=2)
print('evaluating xcep')
#lossxcep, accxcep = xception.evaluate(test_images2, verbose=2)


print("vgg16 accuracy: {:5.2f}%".format(100 * accvgg16))
#print("vgg19 accuracy: {:5.2f}%".format(100 * accvgg19))
#print("resnet50 accuracy: {:5.2f}%".format(100 * accresnet))
#print("cnn accuracy: {:5.2f}%".format(100 * acccnn))

#print("inception accuracy: {:5.2f}%".format(100 * accincp))

#print("xception accuracy: {:5.2f}%".format(100 * accxcep))
