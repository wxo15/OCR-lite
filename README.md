# OCR-lite

A simple Python Flask app to allow users to write a digit on a canvas, and a model will analyze and output the predicted number. Amazon S3 is used for storage of models and parameters.

Web app taken down due to Heroku ending their free plan.


## Models available:
1. Convoluted Neural Network (CNN) built using [TensorFlow's Keras Layers](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer), made up of:
   1. Conv2D(filters=16, kernel_size=(5, 5), padding='same', activation='relu', input_shape=(28, 28, 1))
   2. MaxPooling2D(pool_size=(2, 2))
   3. Conv2D(filters=36, kernel_size=(5, 5), padding='same', activation='relu')
   4. MaxPooling2D(pool_size=(2, 2)),
   5. Dropout(0.25)
   6. Flatten()
   7. Dense(128, activation='relu')
   8. Dropout(0.5)
   9. Dense(10, activation='softmax')
2. sklearn's [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)


<kbd>![screenshots](https://github.com/wxo15/OCR-lite/blob/main/website.gif)</kbd>
