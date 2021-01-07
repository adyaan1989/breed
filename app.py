import streamlit as st
import tensorflow as tf
#from keras.preprocessing.image import ImageDataGenerator
import os
from PIL import Image, ImageOps, ImageEnhance
import cv2
import numpy as np
import io
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.applications.xception import Xception, preprocess_input
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input

import gc
import pandas as pd

####################################################################################################################################
from keras.layers import Lambda, Input, GlobalAveragePooling2D,BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout, GlobalMaxPooling2D, MaxPooling2D, BatchNormalization    
#################################################################################################################################### 

img_size = (331,331,3)

st.set_option('deprecation.showfileUploaderEncoding', False)

labels = pd.read_csv('dog_breeds.csv')
classes = sorted(list(set(labels['breed'])))

##################################################################################################
#Image processing function one
@st.cache
def image_processing(image_data, img_size=(331, 331, 3)):
    size=(331, 331)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.array(image.convert('RGB'))
    image_reshape = np.expand_dims(image, axis=0)
    return image_reshape
# ###################################################################################################################################
def get_features(model_name, model_preprocessor, input_size, data):
       
    input_layer = Input(input_size)
    
    preprocessor = Lambda(model_preprocessor)(input_layer)
    
    base_model = model_name(weights='imagenet', include_top=False,
                           input_shape=input_size)(preprocessor)
       
    
    avg = GlobalMaxPooling2D()(base_model)
    feature_extractor = Model(inputs = input_layer, outputs=avg)
    #Extract Feature
    feature_maps = feature_extractor.predict(data, verbose=1)
    print('Feature maps shape: ', feature_maps.shape)
    return feature_maps


#############################################################################################################################################
model = tf.keras.models.load_model('model_final.hdf5')    
#####################################################################################################################################

def main():
    #st.write("Dog Breeds Prediction")
    #st.write("This is a simple image classification web app to predict the Dog breeds")
    
    activities = ['Image','About']
    choice = st.sidebar.selectbox("Select Activity", activities)

    if choice == 'Image':
        st.subheader("Dog Breed Detection")
        st.info('This is CNN model trained on the following 120 Dog breeds:-')
        st.write(classes)        

        st.set_option('deprecation.showfileUploaderEncoding', False)
        file = st.file_uploader("Please upload an image", type=['jpg','png','jpeg'])

        if file is None:

            st.text("The image has not uploaded")
        
        else:
            image = Image.open(file)
            st.image(image, width=600)
            X = image_processing(image, img_size)

        enhance_type = st.sidebar.radio("Enhance type",['Original','Gray','Contrast','Brightness','Blur'])

        if enhance_type == "Gray":
            st.info('Gray Scaling')
            new_img = np.array(image.convert('RGB'))
            img = cv2.cvtColor(new_img,1)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            st.image(gray)

        elif enhance_type == 'Contrast':
            st.info('Contrasting')
            c_rate = st.sidebar.slider('Contrast',0.5,3.5)
            enhancer = ImageEnhance.Contrast(image)
            output = enhancer.enhance(c_rate)
            st.image(output)

        elif enhance_type == 'Brightness':
            st.info('Brighting')
            c_rate = st.sidebar.slider('Brightness',0.5,3.5)
            enhancer = ImageEnhance.Brightness(image)
            output = enhancer.enhance(c_rate)
            st.image(output)

        elif enhance_type == 'Blur':
            st.info('Bluring')
            new_img = np.array(image.convert('RGB'))
            blur_rate = st.sidebar.slider('Blur',0.5,3.5)
            img = cv2.cvtColor(new_img,1)
            output = cv2.GaussianBlur(img,(11,11), blur_rate)            
            st.image(output)
        #############################################################################
        # task =['Detection']
        # feature = st.sidebar.selectbox('Detection',task)     
        
        if st.button('Deduction'):
            # Extract features using InceptionV3 
            st.info('Let your help AI to Identify the Dog Breed')
            inception_preprocessor = preprocess_input
            inception_features = get_features(InceptionV3, inception_preprocessor, img_size, X)
            st.text('Processing the image........................')
            # st.image(X)
            # st.write( X)                                   
            # ##########################################################################################################################
            ##Extract features using Xception 
            xception_preprocessor = preprocess_input
            xception_feature = get_features(Xception,
                                            xception_preprocessor,
                                            img_size, X)
            
            st.text("It's almost completing..................................")
            # # ##########################################################################################################################
            # # Extract features using InceptionResNetV2 
        
            inc_resnet_preprocessor = preprocess_input
            inc_resnet_feature = get_features(InceptionResNetV2,
                                                inc_resnet_preprocessor,
                                                img_size, X)

            st.text('Moving on last steps..................................')
        ###########################################################################################################################
        #Extract test data features.
        def extact_features(data):
            
            inception_features = get_features(InceptionV3, inception_preprocessor, img_size, data)
            xception_features = get_features(Xception, xception_preprocessor, img_size, data)
            inc_resnet_features = get_features(InceptionResNetV2, inc_resnet_preprocessor, img_size, data)

            final_features = np.concatenate([inception_features,
                                                xception_features,
                                                inc_resnet_features,],axis=-1)
            
            print('Final feature maps test data shape', final_features.shape)
            #deleting to free up ram memory
            del inception_features
            del xception_features
            del inc_resnet_features
            gc.collect()
            return final_features
    # # ##################################################################################################################################
        test_features = extact_features(X)
        st.info('Completed ..........Sorry for delay')
        #st.Image(X)
        pred = model.predict(test_features)
        #################################################################################################################        
        # # #############################################################################################################################
        st.write(f"Predicted Breed: {classes[np.argmax(pred[0])]}")
        st.write(f"Probability of prediction): {round(np.max(pred[0])) * 100} %")

    elif choice == 'About':
        st.subheader("About")
        st.info("Please suggest for correction on E-mail")
        st.text('mlanalytics65@gmail.com')
    

if __name__ == "__main__":
    main()


