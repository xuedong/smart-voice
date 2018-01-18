# Script to calculate audio features.
# Modify MRC to use it!

import os
from subprocess import call

rootDir = 'data/audio/'
dirClean = rootDir+'clean'
dirNoisy = rootDir+'noisy'
dirNoises = rootDir+'noises'

ex = 'src/third-party/audio_features_v1.0/linux/run_compute_audio_features.sh'
MRC = '~/Applications/MATLAB_Runtime/v90'

nb = 0
for dirName, subdirList, fileList in os.walk(dirNoisy):
    print('Found directory: %s' % dirName)
    for fname in fileList:
        if fname[-4:]=='.wav':
            fileCheck = fname[0:-4]
            fileCheck = dirName+'/'+fileCheck+'_Descriptors_Channel_1.txt'
            print(fileCheck)
            if not os.path.isfile(fileCheck):
                print('Processing '+fname)
                print('\t%s' % fname)
                fileName = dirName+'/'+fname
                print(fileName)
                command = ex+' '+MRC+' '+fileName
                print(command)
                call([ex,MRC,fileName])
                nb = nb+1
                # if nb>=1:
                    # break
            else:
                print('Not processing '+fname)

print('Successfully extracted '+str(nb)+' descriptor files.')
