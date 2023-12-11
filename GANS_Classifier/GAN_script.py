import os
import time
import glob
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns 
import imageio
from sklearn.metrics import confusion_matrix
from tensorflow.keras import layers, models
from PIL import Image
from IPython import display

'''HYPERPARAMETERS'''
BATCH_SIZE = 64
'''Amount of training iterations'''
EPOCHS = 100
'''Size of noise vector'''
noise_dim = 100
'''GIF and generator image size'''
num_examples_to_generate = 16
seed = tf.random.normal([num_examples_to_generate, noise_dim])

''' LOADING AND PREPROCESSING
Images resized, normalized, converted to preferred format for input
'''
def load_data(directory):
    data = []
    labels = []
    '''Loop over each folder (letters) in notMNIST_small data folder'''
    for label, letter in enumerate(sorted(os.listdir(directory))):
        letter_dir = os.path.join(directory, letter)
        '''If Statement resolves issue of viewing .DS_Store as a directory. 
        
        Now, skips over any non letter_dir files on path to directory
        '''
        if not os.path.isdir(letter_dir):
            continue
        '''Loop over each image in letters folder'''
        for img_file in os.listdir(letter_dir):
            try:
                img_path = os.path.join(letter_dir, img_file)
                '''Convert image to grayscale'''
                img = Image.open(img_path).convert('L')
                '''Resize to 28 by 28 '''
                img = img.resize((28, 28))
                '''Change PIL image to numpy array'''
                img_array = np.array(img)
                '''Normalize image array values to [-1,1]'''
                img_array = (img_array - 127.5) / 127.5
                '''Additional Channel Dimension for each image to make (28,28,1) shape for discriminator input'''
                img_array = np.expand_dims(img_array, axis=-1)
                data.append(img_array)
                '''Changed label range to adjust to expected input in tensorflow for a 10 

                class problem (range of [0, 9)) given that previously was giving 

                label range of [1, 10)
                '''
                labels.append(label - 1)
            except:
                pass
    data = np.array(data, dtype=np.float64)
    labels=np.array(labels, dtype=np.int64)
    '''Create and return tensorflow dataset from data and labels array for ML models'''
    return tf.data.Dataset.from_tensor_slices((data, labels))

'''GENERATOR MODEL'''
def make_generator_model():
    '''Sequential layering for Neural Network model'''
    model = tf.keras.Sequential()
    '''Dense (fully) connected neural network with 12,544 units (nodes). Input of 100-dimensional noise vector'''
    model.add(layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
    '''Batch Normalization'''
    model.add(layers.BatchNormalization())
    '''LeakyReLU activation function, variation of ReLU'''
    model.add(layers.LeakyReLU())
    '''Reshape output of previous layer'''
    model.add(layers.Reshape((7, 7, 256)))
    '''Transposed Convolution layer, 5 by 5 kernel, stride of 1'''
    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    '''Final transposed convolutional layer, outputs the final image, uses tanh activation function'''
    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    return model

'''DISCRIMINATOR MODEL (similar in structure to generator model)'''
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    '''Added dropout regularization with rate of 0.3'''
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    '''Flattens 3D output from previous convolutional layers in 1D array'''
    model.add(layers.Flatten())
    '''Densely-connected layer for binary classification (real vs fake image)'''
    real_fake_output = layers.Dense(1)(model.output)
    '''Dense layer, 10 units (1 per letter) to classify input image into class'''
    class_output = layers.Dense(10, activation='softmax')(model.output)
    return tf.keras.Model(inputs=model.input, outputs=[real_fake_output, class_output])

'''Loss functions, set from_logits = False to apply softmax function

From_logits= True expects raw, unnormalized scores (logits) in loss function, 

which softmax changed to normalized probabilities
'''
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
class_cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)


'''real_output = discriminators output for real images

fake_output = discriminators output for fake images

real_class_output = discriminators output for classifying real images

real_labels = true labels for real images
'''
def discriminator_loss(real_output, fake_output, real_class_output, real_labels):
    '''Loss for real images'''
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    '''Loss for fake images'''
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    '''Classification loss'''
    class_loss = class_cross_entropy(real_labels, real_class_output)
    '''Total loss of above variables'''
    total_loss = real_loss + fake_loss + class_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

'''Optimizers'''
generator_optimizer = tf.keras.optimizers.Adam(2e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4)

'''Training step performs a single training step for both generator and discriminator. 

Takes in the real images and corresponding labels for real images
'''
def train_step(images, labels):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])
    '''Using Tensorflow 'GradientTape' for automatic differentiation'''
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        '''Passes the real images through the discriminator and returns two outputs from the discriminator'''
        real_output, real_class_output = discriminator(images, training=True)
        fake_output, _ = discriminator(generated_images, training=True)
        '''Quantifiable measures for both generator and discriminator. gen_loss 
        
        quantifies how well the the generator is tricking the discriminator in 
        
        thinking generated images are real. disc_loss calculates how accurate 
        
        the discriminator is in identifying and classifying real images
        '''
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output, real_class_output, labels)

    '''Computes the gradients for both generator and discriminator according to parameter 
    
    inputs above before applying to both generator and discriminator.
    '''
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return gen_loss, disc_loss

'''CLASSIFICATION ACCURACY'''
def calculate_accuracy(model, dataset):
    total = 0
    correct = 0

    for images, labels in dataset:
        _, class_predictions = model(images, training=False)
        predicted_labels = tf.argmax(class_predictions, axis=1)
        correct += tf.reduce_sum(tf.cast(tf.equal(predicted_labels, labels), tf.int32))
        total += images.shape[0]

    return correct / total

'''Generate and save images for GIF creation'''
def generate_and_save_images(model, epoch, test_input, folder='training_images'):
    '''Create a directory of folder images, MAKE IF STATEMENT AS IT LOOPS'''
    if not os.path.exists(folder):
        os.makedirs(folder)

    predictions = model(test_input, training=False)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig(f'{folder}/image_at_epoch_{epoch:04d}.png')
    plt.close(fig)

'''TRAINING FUNCTION, ACCURACY REPORTING'''
def train(dataset, epochs, save_image_folder='training_images'):
    gen_loss_history = []
    disc_loss_history = []
    for epoch in range(epochs):
        start = time.time()

        for image_batch, labels in dataset:
            gen_loss, disc_loss = train_step(image_batch, labels)
            gen_loss_history.append(gen_loss.numpy())
            disc_loss_history.append(disc_loss.numpy())

        '''Calculates and prints classification accuracy after each epoch'''
        accuracy = calculate_accuracy(discriminator, dataset)
        print(f"Epoch: {epoch + 1}, Classification Accuracy: {accuracy.numpy() * 100:.2f}%")

        '''Saves images at the end of each epoch'''
        display.clear_output(wait=True)
        generate_and_save_images(generator, epoch + 1, seed, folder=save_image_folder)

        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    '''Saves images after the final epoch'''
    display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, seed, folder=save_image_folder)
    return gen_loss_history, disc_loss_history

'''CONFUSION MATRIX'''
def generate_confusion_matrix(model, dataset):
    y_true = []
    y_pred = []

    '''Predictions based on true class label and predicted class'''
    for images, labels in dataset:
        _, class_predictions = model(images, training=False)
        predicted_labels = tf.argmax(class_predictions, axis=1)
        y_true.extend(labels.numpy())
        y_pred.extend(predicted_labels.numpy())

    '''Generation of confusion matrix'''
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

'''GIF from images'''
def create_gif(image_folder='training_images', gif_name='dcgan_training_progress.gif'):
    with imageio.get_writer(gif_name, mode='I') as writer:
        filenames = glob.glob(f'{image_folder}/image*.png')
        filenames = sorted(filenames)
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

'''MAIN SCRIPT'''
data_directory = './notMNIST_small'
dataset = load_data(data_directory).shuffle(buffer_size=10000).batch(BATCH_SIZE)

generator = make_generator_model()
discriminator = make_discriminator_model()

'''Train the model and get loss histories'''
gen_loss_history, disc_loss_history = train(dataset, EPOCHS, save_image_folder='training_images')

'''TRAINING HISTORY PLOT FOR D_Loss, G_Loss'''
def plot_training_history(gen_loss_history, disc_loss_history):
    epochs = range(1, len(gen_loss_history) + 1)

    '''Plotting Generator Loss'''
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, gen_loss_history, label='Generator Loss')
    plt.xlabel('Epochs')
    plt.ylabel('G_Loss')
    plt.title('Generator Loss')
    plt.legend()

    '''Plotting Discriminator Loss'''
    plt.subplot(1, 2, 2)
    plt.plot(epochs, disc_loss_history, label='Discriminator Loss')
    plt.xlabel('Epochs')
    plt.ylabel('D_Loss')
    plt.title('Discriminator Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

'''Get and print the final accuracy'''
final_accuracy = calculate_accuracy(discriminator, dataset)
print(f"Final Classification Accuracy: {final_accuracy.numpy() * 100:.2f}%")

'''Save Losses and Accuracy to File'''
with open('training_metrics.txt', 'w') as file:
    for i in range(EPOCHS):
        file.write(f"Epoch {i+1}, Generator Loss: {gen_loss_history[i]}, Discriminator Loss: {disc_loss_history[i]}, Final Classification Accuracy: {final_accuracy.numpy() * 100:.2f}%\n")

'''GIF of the training progress'''
create_gif(image_folder='training_images', gif_name='dcgan_training_progress.gif')

'''Generate and display the confusion matrix'''
generate_confusion_matrix(discriminator, dataset)