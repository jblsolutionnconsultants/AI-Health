#!/usr/bin/env python
######################################################################
##  Copyright (c) 2019 Jayanath Liyanage <jayanath1987@gmail.com>   ##
##  All rights reserved.                                            ##
##  Licensed under the New BSD License                              ##
##  (http://www.freebsd.org/copyright/freebsd-license.html)         ##
######################################################################
import PIL.Image
import os
import numpy as np
import cv2
import pickle
import random

try:
    from Tkinter import *
    import tkFileDialog as filedialog
except ImportError:
    from tkinter import *
    from tkinter import filedialog
    from tkinter import messagebox
    from pprint import pprint
    from var_dump import var_dump
    import os
import PIL.ImageTk
data=[]
target=[]
img_array=[]
test_size = 0.1
B1=""


class App(Frame):
    def chg_image(self):
        if self.im.mode == "1": # bitmap image
            self.img = PIL.ImageTk.BitmapImage(self.im, foreground="white")
        else:              # photo image
            self.img = PIL.ImageTk.PhotoImage(self.im)
        #self.la.config(image=self.img, bg="#000000",width=self.img.width(), height=self.img.height())
        self.la.config(image=self.img, bg="#000000",width=500, height=500)
        self.lbl_text.set("")

    def load_data(self):
        with open('my_dataset1.pickle', 'rb') as data1:
            dataset1 = pickle.load(data1)
            dataset1 = np.array(dataset1)
        img_array.append(dataset1)

        with open('my_dataset2.pickle', 'rb') as data2:
            dataset2 = pickle.load(data2)
            dataset2 = np.array(dataset2)
        img_array.append(dataset2)

    def builddata(self):
        for i in img_array:
            for j in i:
                data.append(j[:65535])
                target.append(j[65536])

    def open(self):
        filename = filedialog.askopenfilename(filetypes = (("jpeg files","*.jpeg"),("jpg files","*.jpg")))
        print(filename)
        self.fpath=filename
        if filename != "":
            self.im = PIL.Image.open(filename)
        self.chg_image()
        self.num_page=0
        self.num_page_tv.set(str(self.num_page+1))


    def seek_prev(self):
        self.num_page=self.num_page-1
        if self.num_page < 0:
            self.num_page = 0
        self.im.seek(self.num_page)
        self.chg_image()
        self.num_page_tv.set(str(self.num_page+1))

    def seek_next(self):
        self.num_page=self.num_page+1
        try:
            self.im.seek(self.num_page)
        except:
            self.num_page=self.num_page-1
        self.chg_image()
        self.num_page_tv.set(str(self.num_page+1))
    def resize_image(self):
        #print("resize image")
        #self.load_data()  # data load from pickle
        #self.builddata()  # data,target seperator
        #entries = os.listdir('data/')
        #for e in entries:
         #   with open(e, 'r+b') as f:

        #img = cv2.imread('02840063123.jpeg', cv2.IMREAD_UNCHANGED)
        print(self.fpath)

        img = cv2.imread(self.fpath, cv2.IMREAD_UNCHANGED)
        cover = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        #print(cover)
        new_img_array = np.array(cover)
        new_img_array = cv2.cvtColor(new_img_array, cv2.COLOR_BGR2GRAY)

        new_img_array = cv2.cvtColor(cover, cv2.COLOR_BGR2GRAY)
        new_img_array = np.resize(new_img_array, (1, 65535))
        #img_array = np.append(img_array, [0])
        #print(new_img_array[:65536])
        #print(new_img_array[0][65535])
        #print(new_img_array.shape)

        # load the model from disk
        filename = 'finalized_model_SVM.sav'
        loaded_model = pickle.load(open(filename, 'rb'))
        #print(loaded_model)
        #img_array = np.resize(img_array[0:65535], (1, 65535))
        #score = loaded_model.score(img_array[0:65535], [1])
        #print("Test score: {0:.2f} %".format(100 * score))
        Ypredict = loaded_model.predict(new_img_array)
        print(Ypredict[0])
        return Ypredict[0]
        #print('Actual       :', img_array)



        #pprint(globals())
        #pprint(locals())
        ##from sklearn.model_selection import train_test_split
        ##train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=test_size)
        ##from sklearn.svm import SVC
        ##clsfr = SVC(kernel='linear')  # support vector classifier
        ##clsfr.fit(train_data, train_target)
        #print(test_data[0],"asdasdasd")
        ##a = np.array(new_img_array[0:65535])
        ##b= np.resize(a, (1, 65535))

        #results = clsfr.predict(test_data)
        ##results = clsfr.predict([b[0]])
        ##print('Actual       :', test_target)
        ##print('SVM Predicted:', results)
        ##from sklearn.metrics import accuracy_score
        ##accuracy = accuracy_score(test_target, results)
        ##print('SVM Accuracy :', accuracy)

    def predict(self):

        self.lbl_text.set("Predicted")
        val = self.resize_image()

        if(val==1):
            #messagebox.showinfo("Predicted      ©JBLSolutions", "Cancer Negative      ")
            messagebox.showinfo("Predicted      ©JBLSolutions", "Male      ")
        else:
            #messagebox.showerror("Predicted      ©JBLSolutions", "Cancer Positive, Meet a Doctor")
            messagebox.showerror("Predicted      ©JBLSolutions", "Female")


    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master.title('Artifical Intelligence Cancer Predictor                ©JBLSolutions')
        self.img_array=[]
        self.num_page=0
        self.num_page_tv = StringVar()
        self.lbl_text = StringVar()
        self.filename = StringVar()
        self.fpath = StringVar()



        fram = Frame(self)
        Button(fram, text="Open File", command=self.open).pack(side=LEFT)
        #Button(fram, text="Prev", command=self.seek_prev).pack(side=LEFT)
        #Button(fram, text="Next", command=self.seek_next).pack(side=LEFT)
        Button(fram, text="Predict", command=self.predict).pack(side=LEFT)

        #Label(fram, textvariable=self.num_page_tv).pack(side=LEFT)
        Label(fram, textvariable=self.lbl_text).pack(side=LEFT)
        fram.pack(side=TOP, fill=BOTH)

        self.la = Label(self)
        self.la.pack()
        self.la.config( bg="#000000", width=100, height=30) #Fixed Width


        self.pack()

if __name__ == "__main__":
    app = App(); app.mainloop()
