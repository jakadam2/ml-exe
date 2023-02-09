from tensorflow import keras
from image_loader import load_one_image_tocnn
model = keras.models.load_model('digit_recognizer')
for i in range(10):
    print(i,': ',round(model.predict_on_batch(load_one_image_tocnn('img/' + '0.png'))[0][i]*100,2),'%')

