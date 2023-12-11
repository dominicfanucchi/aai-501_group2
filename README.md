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

### Explain Multinomial Logistic Regression Here
### Explain CNN Here
### Explain GAN Here


## Project Description (Erase when done)
Discuss the details of project overview. Description your selected dataset, such as data source, number of variables, size of dataset, etc. Include data dictionary, if available.  Provide questions and hypothesis that you are exploring. What specific data analysis, visualization, and modeling work are you using to solve the problem? What roadblocks and challenges are you facing? etc. 

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
   

## Results

## Acknowledgments
We would like to express our sincere gratitude to Professor David Friesen, M.S. who taught the AAI-501-02 Course, and Yaroslav Bulatov for putting together this dataset. 

##References 

## License
This dataset is licensed under a [CC0 1.0 DEED license](https://creativecommons.org/publicdomain/zero/1.0/legalcode.en) - see the [Creative Commons](https://creativecommons.org/publicdomain/zero/1.0/legalcode.en) website for details.
