# Embedded-Systems
Image Compression on Embedded System Readme
Problem Statement
Implement learned image compression on a small embedded system in a resource-constrained environment. Develop a custom model for learned image compression and compare its performance with standard models like JPEG, JPEG2000, BPG, etc. Provide a detailed explanation of the preprocessing stage, layers in the model, their importance, and conclusions drawn from the results.

# Importing Libraries
Imports necessary modules and functions from various libraries including Keras (Model, Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, BatchNormalization), Adam optimizer, MeanSquaredError loss, Matplotlib, NumPy, Pandas, and OpenCV.

# Dataset Visualization
Visualizes the dataset to understand the nature of images and data distribution.
![image](https://github.com/JyotiDhayal/Embedded-Systems/assets/112848090/31c6f4c0-eb7a-4b1e-b4c8-c7b4afcfc7cf)

# Preprocessing
Converting Color Channels: Converts the images from BGR to RGB format using OpenCV's cv2.cvtColor() function.
Resizing Images: Resizes the images to a fixed size of 400x400 pixels to ensure uniform dimensions.
Normalizing Pixel Values: Scales the pixel values to the range [0, 1] by dividing by 255.0.
Storing and Splitting Preprocessed Images: Stores preprocessed images in a list and splits the dataset into training, validation, and testing sets with an 80-10-10 split ratio.
Autoencoder Architecture
Defines the architecture for an improved convolutional autoencoder model for image compression.

# Model Architecture
Encoder: Consists of convolutional layers followed by batch normalization and max-pooling operations to reduce the spatial dimensions and capture essential features.
Decoder: Comprises convolutional layers followed by batch normalization and up-sampling operations to reconstruct the original image from the compressed representation.
Output: The output layer consists of a convolutional layer with a sigmoid activation function to generate the reconstructed image with three channels (RGB).
Model Compilation
Compiles the model using the Adam optimizer and Mean Squared Error (MSE) loss function.

# Training Model
Trains the defined model using training data, specifying the number of epochs, batch size, and providing validation data to monitor performance.

# Plotting Results
Creates a figure with subplots displaying the original and compressed images for comparison.
![image](https://github.com/JyotiDhayal/Embedded-Systems/assets/112848090/641a95b6-ec25-4c32-bdd1-4025def77c68)


# Saving Compressed Image
Saves the compressed image to a file using OpenCV's imwrite function.

# Calculating Compression Ratio
Calculates the compression ratio by dividing the size of the original image by the size of the compressed image and prints the result.

# Standard JPEG Algorithm
Sets compression quality for saving an image in JPEG format, saves the image, and calculates the compression ratio between the PNG and JPEG images.

# Conclusion
Based on the comparison of compression ratios, it is concluded that the custom model performs better than the standard JPEG algorithm.
![image](https://github.com/JyotiDhayal/Embedded-Systems/assets/112848090/fa10b526-044b-4a43-9a01-5b795e354db2)

