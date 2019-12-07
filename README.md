# 501_ai_assignment

I have not changed any of the command line arguments or file names in any of the scripts, so they can all be run the same as the provided versions on the course website.

## Part 1 Report

To improve the accuracy of this model, I did the following:
* Added more dense layers, as one layer is not optimal.
* Changed number of units in top two layers to 128.
* Changed layer activation types (relu for input layers, softmax for output).
* Changed model optimizer to Adam, as it seems to perform better than sigamoid.

I made these changes based on some quick research as to which layers are optimal for training the model to recognize images.

## Part 2 Report

To get the prediction scripts working and to train the notMNIST model, I did the following:
* Added training for notMNIST similar to MNIST training
* Added code to save the models into .h5 files
* In the prediction scripts, I loaded the model(s) and did the prediction by adding the following lines of code to the predict method:
```
prediction = model.predict(img)
prediction = prediction[0]
predicted_label = np.argmax(prediction)
```
I found two predictions that were predicting incorrectly to base further improvements on:
* for MNIST data, this I used test image 8 (predicts a 6, but is actually a 5).
* for notMNIST data, I used test image 1 (predicts a G, but is actually an A).

I then added convolution layers to both models. Adding convolution layers improved accuracy in both models tremendously
* for MNIST data, the test image prediction for image 8 is now correct (image 8 now correctly predicts 5).
* for notMNIST data, the test image prediction for image 1 is actually now more wrong, however, the model is still much more accurate.

Adding convolution layers broke the prediction scripts, however, I could only use predict_test reliably, as I could not get images to correctly work on my machine. Thus, I only fixed predict_test to get it to work with the new layers.


## Part 3 Report

For this part, I primarily followed the tutorial for importing CSV. I didn't have time to experiment with convolution very much, however even without convolution I was able to get the sample to 67% accuracy. It does seem to overfit on some tests I ran, but I know this could be combatted a bit by implementing some convolution.