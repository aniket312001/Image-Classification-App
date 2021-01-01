"""
    By Aniket Anil Chavan 
"""


from flask import Flask,render_template,request,send_from_directory,url_for
import cv2
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Activation


image_shape = (150,150,3)

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3,3),input_shape=image_shape, activation='relu',))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape=image_shape, activation='relu',))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=64, kernel_size=(3,3),input_shape=image_shape, activation='relu',))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(50,activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(6,activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.load_weights('static/image_classication_6.h5')





app = Flask(__name__)

COUNT = 0
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/pred',methods=['POST','GET'])
def predictions():
    global COUNT
    img = request.files['img']  # loading img

    img.save(f'static/{COUNT}.jpg')    # saving img
    img_arr = cv2.imread(f'static/{COUNT}.jpg')    # converting into array

    img_arr = cv2.resize(img_arr, (150,150))
    img_arr = img_arr / 255.0
    img_arr = img_arr.reshape(1, 150,150,3)
    ans = model.predict(img_arr)  # it will give class
    classes = ['buildings', 'forest', 'glacier', 'mountain', 'sea','street']

    res = ans[0]==max(ans[0])
    j=0;
    for i in res:
        if i:
            predict = classes[j]
        j = j+1

    COUNT += 1

    return render_template('prediction.html',data=predict)


@app.route('/load_img')
def display_image():
    global COUNT
    return send_from_directory('static', f"{COUNT-1}.jpg")




if __name__ == "__main__":
    app.run(debug=True)

