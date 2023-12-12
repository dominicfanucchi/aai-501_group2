# Group 2 Final Project
This project is a part of the AAI-501 course in the Applied Artificial Intelligence Program at the University of San Diego (USD).

**-- Project Status: Active**

- ### Partner(s)/Contributor(s)
   * Dominic Fanucchi
   * Carlo Sanchez
   * Hani Jandali

## Installation
To use this project, first clone the repo on your device using the command below:
```
git init
git clone https://github.com/dominicfanucchi/aai-501_group2.git
```

## Project Objective
Image classification is a widely recognized section of machine learning which often extends into the fields of deep learning and artificial intelligence. In this project, each team member applied a unique image classifying algorithm trained and tested on the small notMNIST dataset. Specifically, the purpose of this project is to classify different letters, ranging from A to J, into their own category (class) across all stylistic forms provided in the dataset, with a further ability to predict the digit from the image with a relatively significant accuracy among other potential performance measures. 

## About the Dataset
The MNIST dataset is one of the best known image classification problems out there, and a veritable classic of the field of machine learning. This dataset is more challenging version of the same root problem: classifying letters from images. This is a multiclass classification dataset of glyphs of English letters A - J.

notMNIST_large.zip is a large but dirty version of the dataset with 529,119 images, and notMNIST_small.zip is a small hand-cleaned version of the dataset, with 18726 images. The dataset was assembled by Yaroslav Bulatov, and can be obtained on his [blog](https://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html). According to this blog entry there is about a 6.5% label error rate on the large uncleaned dataset, and a 0.5% label error rate on the small hand-cleaned dataset.

The two files each containing 28x28 grayscale images of letters A - J, organized into directories by letter. notMNIST_large.zip contains 529,119 images and notMNIST_small.zip contains 18726 images.

## Approach
Image classification on the notMNIST dataset was approached from both a machine learning and deep learning implementation. The specific algorithms and networks used were as follows: 

  * Machine Learning
    * Multinomial Logistic Regression
  * Deep Learning Neural Networks
    * Convolutional Neural Network (C.N.N)
    * Semi-Supervised Generative Adversarial Network (G.A.N)

These algorithms and networks were implemented through Python and Jupyter Notebooks. 

 - ### Imports and Libraries
   * Numpy 
   * Pandas 
   * Matplotlib / Pyplot
   * Scipy Stats
   * Seaborn 
   * Tensorflow
   * Keras 
   * Glob 
   * Sklearn
   * IPYthon
   * Imageio 
   * Time
   * Glob
   * OS

## Models
  ### Multinomial Logistic Regression
  Multinomial Logistic Regression is a statistical method used for classification often in scenarios requiring more than two categories in which "it predicts the probability of one of three or more possible outcomes" (Saini, 2023). To apply the algorithm to the notMNIST dataset, we converted each image to grayscale and flattened into a one-dimensional array representation before training on Scikit-Learn's built-in Logistic Regression function. 

  ### CNN 
  Originally developed by Yann LeCun, the Convolutional Neural Network, or CNN, was originally designed for image recognition, specifically handwritten digits (ProjectPro). Incorporating this neural network onto the notMNIST dataset, we mapped image classes to integers, performing one-hot encoding for image labels. Using the Keras API, we reshaped and normalized our data before feeding it into a sequential model with an initial convolutional layer, a pooling layer, and a flatten layer with 128 neurons, before applying a 15% dropout and entering a final 10 neuron layer representing our 10 classes. 

  ### Explain GAN Here
  The semi-supervised deep convolutional generative adversarial network (DCGAN) was originally developed by Ian Goodfellow and Tim Salimans (Goodfellow, 2016). Under the context of this project, we sought to classify each instance of the dataset into their respective class out of ten possibilities while simultaneously generating artificial instances of those images and capturing them. The model was created using Tensorflow and Keras, processing the data into gray scaled (28,28,1) instances before fed into the generator model, layered sequentially, with convolutional and batch normalization layers before being fed into the discriminator network composed of sequential convolutional and dropout layers. 


## Results

The overall results were incredibly positive and were as follows (by measure of classification accuracy): 

 - Multinomial Regression: 86.14%
 
 - Convolutional Neural Network (10 Epochs): 92.34%

 - Semi-Supervised DCGAN (100 Epochs): 95.96%

![Gif of Semi-Supervised DCGAN](https://github.com/dominicfanucchi/aai-501_group2/blob/main/GANS_Classifier/Run_Two/Run_2.gif)

## References

Bulatov, Yaroslav (2011). notMNIST dataset. [Data set]. Yaroslav Bulatov. https://yaroslavvb.blogspot.com/2011/09/notmnist-dataset.html

Goodfellow, I. J., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farkley, D., Ozair, S., Courville,
  A., Bengio, Y., (2014). Generative Adversarial Nets. Advances in Neural Information 
  Processing Systems, 27. https://doi.org/10.48550/arXiv.1406.2661

Goodfellow, I., Salimans, T., Zaremba, W., Cheung, V., Radford, A., Chen, X., (2016). Improved
  Techniques for Training GANs. 30th Conference on Neural Information Processing
  Systems (NIPS 2016). https://doi.org/10.48550/arXiv.1606.03498

Lex Friedman. (2019, April 18). Ian Goodfellow: Generative Adversarial Networks (GANs) | Lex 
  Friedman Podcast # 19 [Video]. YouTube. https://www.youtube.com/watch?v=Z6rxFNMGdn0&t=3143s&ab_channel=LexFridman

Lubaroli (2017). notMNIST dataset. [Data set]. Kaggle. 
  https://www.kaggle.com/datasets/lubaroli/notmnist 

ProjectPro. (n.d.). Introduction to Convolutional Neural Networks Architecture. 
  https://www.projectpro.io/article/introduction-to-convolutional-neural-networks-algorithm-architecture

Russell, S., & Norvig, P. (2021). Artificial intelligence: A modern approach (4th ed.). Pearson.

Saini, Anshul. (2023, October 30). A Beginner’s Guide to Logistic Regression. Analytics Vidhya.     https://www.analyticsvidhya.com/blog/2021/08/conceptual-understanding-of-logistic-regression-for-data-science-beginners/#h-what-is-logistic-regression

Tripathi, Mohit. (2023, May 1). Image Processing using CNN: A beginners guide. Analytics Vidhya.
  https://www.analyticsvidhya.com/blog/2021/06/image-processing-using-cnn-a-beginners-guide


## Acknowledgments
We would like to express our sincere gratitude to Professor David Friesen, M.S. who taught the AAI-501-02 Course, and Yaroslav Bulatov for putting together this dataset. 


## License
This dataset is licensed under a [CC0 1.0 DEED license](https://creativecommons.org/publicdomain/zero/1.0/legalcode.en) - see the [Creative Commons](https://creativecommons.org/publicdomain/zero/1.0/legalcode.en) website for details.
