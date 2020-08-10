from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import cv2
import train
import argparse
import numpy as np

def main(args):
    """ 
    Description - Main function
    :type args: list
    :param args: List of arguments to parse
    """
    
    print("Enter model type to run inference - 1. Densenet, 2. Inception, 3. Pruned, 4. TFlite")
    model_type = int(input())
    if model_type == 1:
        model_name = "densenet"
    elif model_type == 2:
        model_name = "inception"
    elif model_type == 3:
        model_name = "pruned"
    elif model_type == 4:
        model_name = "quantized"
    else:
        raise Exception("Incorrect model name")

    img = load_img(args.imagepath)
    img = img_to_array(img)
    img = cv2.resize(img,(224,224))
    test_image = np.expand_dims(img, axis=0).astype(np.float32)

    pred_label = {0:"Negetive Pneumonia", 1:"Positive Pneumonia"}

    if (model_name == "densenet" or model_name == "inception"):
        # weights.hdf5 or model.h5 is the expected input here
        model = train.build_model(model_name)
        print("Enter path to saved weights file (.hdf5 or .h5 format) ")
        weightsfile = str(input())
        model.load_weights(weightsfile)
        prediction = model.predict(test_image)
        print(pred_label[np.argmax(prediction)])

    elif model_name == "pruned":
        #model.h5 is the expected input here
        print("Enter path to saved weights file (.h5 format) ")
        weightsfile = str(input())
        model = tf.keras.models.load_model(weightsfile)
        model.compile( loss='categorical_crossentropy', optimizer=Adam(lr=1e-5), metrics=['accuracy'])
        prediction = model.predict(test_image)
        print(pred_label[np.argmax(prediction)])

    elif model_name == "quantized":
        print("Enter path to tflite model file (.tflite format)")
        weightsfile = str(input())
        interpreter = tf.lite.Interpreter(model_path=str(weightsfile))
        interpreter.allocate_tensors()
        input_index = interpreter.get_input_details()[0]["index"]
        output_index = interpreter.get_output_details()[0]["index"]

        interpreter.set_tensor(input_index, test_image)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_index)
        print(pred_label[np.argmax(prediction)])

    else:
        raise Exception("Incorrect model name")

    print("Done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('imagepath', metavar="image", type=str, help="Image to perform detection")
    args = parser.parse_args()
    main(args)