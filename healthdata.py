import os
from PIL import Image
from docutils.nodes import image
#from resizeimage import resizeimage
import numpy as np
import cv2
import pickle


data=[]
target=[]
img_array=[]
test_size = 0.1


def resize():
    #entries = os.listdir('data/')
    #entries = os.listdir('/media/Acer/JBL/Data/201905/F/')
    entries = os.listdir('/media/jayanath/Data/RandD/ai/my/my_project/RealData/Data/1/')
    print(entries)
    for e in entries:
        #with open('data/'+e, 'r+b') as f:
        with open('/media/jayanath/Data/RandD/ai/my/my_project/RealData/Data/1/'+e, 'r+b') as f:
           with Image.open(f) as image:
            #cover = resizeimage.resize_cover(image, [256, 256])
            #cover = cv2.resize_cover(image, (256, 256), interpolation = cv2.INTER_AREA)
            cover = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
            cover.save('/media/jayanath/Data/RandD/ai/my/my_project/RealData/converted/1/'+e, image.format)
    print('Resize images are saved to converted dir.')

def np_male_array():
    entries = os.listdir('converted/1/')
    for e in entries:
        with open('converted/1/'+e, 'r+b') as f:
           with Image.open(f) as image:
            img_array1 = np.array(image)
            img_array1 = cv2.cvtColor(img_array1, cv2.COLOR_BGR2GRAY)
            img_array1 = np.reshape(img_array1, (1, 65536))
            #img_array1 = img_array / 65536.0 * 15.0
            img_array1 = np.append(img_array1,[1])
            img_array.append(img_array1)

    with open('my_dataset11.pickle', 'wb') as output:
        #pickle.dump(img_array, output)
        pickle.dump(img_array, output, protocol=2)
    #print(img_array)
    #print(img_array.__sizeof__())

def np_female_array():
    entries = os.listdir('converted/2/')
    for e in entries:
        with open('converted/2/'+e, 'r+b') as f:
           with Image.open(f) as image:
            img_array2 = np.array(image)
            img_array2 = cv2.cvtColor(img_array2, cv2.COLOR_BGR2GRAY)
            img_array2 = np.reshape(img_array2, (1, 65536))
            #img_array1 = img_array / 65536.0 * 15.0
            img_array2 = np.append(img_array2,[2])
            img_array.append(img_array2)
            #print(img_array2)
    #np.savetxt('my_dataset2.txt', img_array)
    with open('my_dataset22.pickle', 'wb') as output:
        pickle.dump(img_array, output, protocol=2)
    #print(img_array)
    #print(img_array.__sizeof__)


def load_data():
    with open('my_dataset1.pickle', 'rb') as data1:
        dataset1 = pickle.load(data1)
        dataset1 = np.array(dataset1)
    img_array.append(dataset1)

    with open('my_dataset2.pickle', 'rb') as data2:
        dataset2 = pickle.load(data2)
        dataset2 = np.array(dataset2)
    img_array.append(dataset2)


def builddata(img_array):
    for i in img_array:
        for j in i:
            data.append(j[:65535])
            target.append(j[65536])

    #print(data[55])

def KNN():
    from sklearn.model_selection import train_test_split
    train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=test_size)
    from sklearn.neighbors import KNeighborsClassifier  # load KNN classifer
    clsfr = KNeighborsClassifier(n_neighbors=1)  # KNN classifier is loaded to clsfr
    clsfr.fit(train_data, train_target)  # training the ML algorithm(KNN)
    results = clsfr.predict(test_data)
    print('Actual       :', test_target)
    print('KNN Predicted:', results)
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(test_target, results)
    print('KNN Accuracy :', accuracy)


def SVM():
    from sklearn.model_selection import train_test_split
    train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=test_size)
    from sklearn.svm import SVC
    clsfr = SVC(kernel='linear')  # support vector classifier
    clsfr.fit(train_data, train_target)
    results = clsfr.predict(test_data)
    print('Actual       :', test_target)
    print('SVM Predicted:', results)
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(test_target, results)
    print('SVM Accuracy :', accuracy)

def DT(): #Decision Tree
    from sklearn.model_selection import train_test_split
    train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=test_size)
    from sklearn import tree
    clsfr = tree.DecisionTreeClassifier()
    clsfr.fit(train_data, train_target)
    results = clsfr.predict(test_data)
    print('Actual       :', test_target)
    print('DT Predicted :', results)
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(test_target, results)
    print('DT Accuracy  :', accuracy)


resize() # 1) Resize images are saved to converted dir
#np_male_array()  # 2) Load male images to numpy array
#np_female_array()
##load_data() # data load from pickle
##builddata(img_array) # data,target seperator
##KNN() #K Nearest Neighbors
##SVM() #Support Vector Machine
##DT() #Decision Tree
