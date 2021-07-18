# Real-time LIS recognition with Deep Learning and OpenCV 


This repo contains the code for training and evaluating the models used in the paper "Real-time Italian Sign Language Recognition
with Deep Learning" (2021). Its aim is the application of deep learning and fine-tuning techniques to build an automatic recognition system for the Italian Sign Language (LIS). More specifically, the main goal is a real-time image recognition system capable of accurately identifying the letters of the alphabet of the LIS provided by a user in a Human Computer Interaction (HCI) framework by means of the Python’s Open Source Computer Vision library and Keras’ VGG19, a convolutional network architecture applied for large-scale image and video recognition. In addition to testing the performance of different classification architectures with slightly modified parameters, this work constitutes a novel step towards the application of automatic image recognition techniques with the recently acknowledged LIS and the only LIS alphabet open-source dataset (https://drive.google.com/file/d/1AFcb2VnGCn2OslIlB6kFpVDAS3PMNqvs/view). 
This project may not only play a role in the interpretation and learning of the Italian Sign Language, encouraging its spread and study, but also in the inclusion of hearing-impaired individuals in society and in the language research domain.

If you have any questions about this code or have problems getting it to work, please send me an email at ```vschmalz@unibz.it```.


## Dependencies
Tensorflow, keras, numpy, opencv, pandas, matplotlib

## Training and evaluation
First, change the ```DATASET_FOLDER``` and/or ```train\valid\test_data_dir``` to point to where the sign language image Data data and/or LIS data are stored on your computer/Drive.

```models\CNN_LIS```:_ To train the CNN model on a dataset 

```models\deep_VGG19```:_ To fine-tune the pre-trained VGG19 deep neural model on a dataset 

## Real-time HCI testing 
First, change the ```model_path``` to point to where the pre-trained model is stored on your computer/Drive.

```opencv_LIS```:_ To test one of the pre-trained models on real-time LIS signs  




## Citation
If you find this repo or our Fluent Speech Commands dataset useful, please cite it:

- Schmalz, Veronica Juliana (2021): Real-time LIS recognition with Deep Learning and OpenCV, https://github.com/VeroJulianaSchmalz/LIS-Recognition-with-Deep-Learning-/
