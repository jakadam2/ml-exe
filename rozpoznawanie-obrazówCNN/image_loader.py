from PIL import Image
import numpy as np
# this functions load common PNG file and convert them ready for use by CNN 
def load_one_image_tocnn(path,width = 28):
    image = Image.open(path)
    pic = [[(255 - list(image.getdata())[width*j + i][0])/255.0 for i in range(width)] for j in range(width)]
    pic = np.array(pic)
    pic = pic.reshape((1, width, width, 1))
    return pic

def load_many_images_tocnn(list_of_paths,width = 28):
    images = []
    for path in list_of_paths:
        image = Image.open(path)
        pic = [[list(image.getdata())[width*i + j][0]/255.0 for i in range(width)] for j in range(width)]
        images.append(pic)
    images = np.array(images)
    images = images.reshape(images.shape[0],width,width,1)
    return images


