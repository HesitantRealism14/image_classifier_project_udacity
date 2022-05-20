from PIL import Image
import numpy as np
import json
import tensorflow as tf
import tensorflow_hub as hub
import argparse

# Input the image path to the image you want to classify
# (you can choose to specify the model path, top k predictions, or label map path as well)
# and get the predictions and the probabilities associated with each prediction
def predict(image_path, model='./my_model.h5', top_k=5, category_names='./label_map.json'):
    with open(category_names, 'r') as f:
        class_names = json.load(f)
    image_size=224
    image = np.asarray(Image.open(image_path))
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (image_size, image_size))
    image /= 255
    processed_image = image.numpy()

    model = tf.keras.models.load_model(model,custom_objects={'KerasLayer':hub.KerasLayer})
    probs = np.sort(model.predict(np.expand_dims(processed_image, axis=0)))[0][::-1]
    classes = np.add(np.argsort(model.predict(np.expand_dims(processed_image, axis=0)))[0][::-1], 1)
    classes_str = np.array(classes[:top_k], dtype='str')
    classes_return = []
    for class_str in classes_str:
        classes_return.append(class_names[class_str])
    return probs[:top_k], classes_return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('arg1')
    parser.add_argument('--model')
    parser.add_argument('--top_k')
    parser.add_argument('--category_names')

    args = parser.parse_args()
    image_path = args.arg1

    if not args.model:
        model = './my_model.h5'
    else:
        model = args.model

    if not args.top_k:
        top_k = 5
    else:
        top_k = int(args.top_k)

    if not args.category_names:
        category_names = './label_map.json'
    else:
        category_names = args.category_names

    args = parser.parse_args()

    probs, classes = predict(image_path, model, top_k, category_names)

    print(probs)
    print(classes)
