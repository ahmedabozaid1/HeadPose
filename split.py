import os
import numpy as np
import shutil

rootdir = '../Data/'  # path of the original folder

classes = ['focus', 'left', 'right']

for i in classes:

    os.makedirs(rootdir + '/Train/' + i)

    os.makedirs(rootdir + '/Test/' + i)

    source = rootdir + '/' + i

    allFileNames = os.listdir(source)

    np.random.shuffle(allFileNames)

    test_ratio = 0.25

    train_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                               [int(len(allFileNames) * (1 - test_ratio))])

    train_FileNames = [source+'/' + name for name in train_FileNames.tolist()]
    test_FileNames = [source+'/' + name for name in test_FileNames.tolist()]

    for name in train_FileNames:
        shutil.copy(name, rootdir + '/Train/' + i)

    for name in test_FileNames:
        shutil.copy(name, rootdir + '/Test/' + i)
