1. Organize Your Data in directories
   1. Rename to C/DP_001 and convert to jpg
      1. image_converter.py
   2. Data size is limited
      1. Using AI and Image Generation
         1. DALL-E not good enough
         2. https://github.com/pgaston/ditherusd
            1. Should I create my tool for this? It could be beneficial to many others too.
      2. Data Augmentation
         1. ~~Using Keras/TensorFlow's ImageDataGenerator~~
            1. Maybe didn't do it right? https://www.datacamp.com/tutorial/complete-guide-data-augmentation
         2. RandomFlip and RandomRotation from keras.layers
            1. Load data
2. Labeling
   1. Manual labeling: simply place each image in its respective folder.
3. Splitting the Dataset
   1. 70% training, 15% validation, and 15% testing
      1. keep subfolders, control and positive
4. Train
   1. create_model:  convolutional neural network (CNN) model
   2. 

<!-- This is more like my experiment note than a README file -->