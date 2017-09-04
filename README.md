# How do THEY do it ?

This is a repository contains following things:

 * Tensorflow Android app 
 * Everything about Tensorflow
 * Interesting applications of Artificial Intelligence
 * Image Net
 * InceptionV3 model
 * Training your own Image Classifier
 * few more interesting things...! 

### Tensorflow:
* TensorFlow is an open source software library for numerical computation using graphs. 
* Nodes in the graph represent mathematical operations, while  edges represent the multidimensional data arrays (tensors) communicated between them.
* The flexible architecture allows you to deploy computation to one or more CPUs or GPUs in a desktop, server, or mobile device with a single API. 
* TensorFlow was originally developed by researchers and engineers working on the Google Brain Team within Google's Machine Intelligence research organization for the purposes of conducting machine learning and deep neural networks research, but the system is general enough to be applicable in a wide variety of other domains as well. 

For further details on Tensorflow visit here [https://www.tensorflow.org/](https://www.tensorflow.org/)

### Artificial Intelligence Experements:

here are some of the very interesting AI exprements, see and build one yourself. 
  visit here [https://experiments.withgoogle.com/ai](https://experiments.withgoogle.com/ai)

### Image Net:
ImageNet is an image database organized according to the WordNet hierarchy (currently only the nouns), in which each node of the hierarchy is depicted by hundreds and thousands of images. Currently it have an average of over five hundred images per node. ImageNet is a useful resource for researchers, educators, students and all of you who share our passion for pictures.
visit here [http://www.image-net.org/](http://www.image-net.org/)

### Inception Model:

Inception was developed at Google to provide state of the art performance on the ImageNet Large-Scale Visual Recognition Challenge and to be more computationally efficient than its competitor architectures. However, what makes Inception exciting is that its architecture can be applied to a whole host of other learning problems in computer vision.
<br/> visit here: [Go deeper, tensorflow and inception](https://medium.com/initialized-capital/we-need-to-go-deeper-a-practical-guide-to-tensorflow-and-inception-50e66281804f)

### Training your own Image Classifier:
#### 1. Gather your training dataset:
* Firstly you have to gather your own dataset, How to do it? <br/> if you dont have a data set and want one you are at the right place. Here is the trick.<br/> Can you not make a dataset from search result of google images?<br/> Exactly, you can. Here's how you will do it.<br/>Visit Here [https://32hertz.blogspot.in/2015/03/download-all-images-from-google-search.html](https://32hertz.blogspot.in/2015/03/download-all-images-from-google-search.html)

* Now cleaning your dataset seems to be a easy job.
  
* Place the Dataset in a folder with different classes as different folders in the dataset. <br/>(if you have flowers as a dataset then the classes will contain<br/> /Daisy<br/>/roses<br/>/sunflowers <br/> different floders with desired images)

#### 2. Investigate the training script:
The retrain script is part of the [tensorflow repo](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/retrain.py), but it is not installed as part of the pip package. So for simplicity I've included it in this repository. copy the retrain.py fie to the working directory. You can run the script using the python command. Take a minute to skim its "help".

```
python -m scripts.retrain -h
```
#### 3. Run the Training:
As noted in the introduction, Imagenet models are networks with millions of parameters that can differentiate a large number of classes. We're only training the final layer of that network, so training will end in a reasonable amount of time.

Start your retraining with one big command (note the --summaries_dir option, sending training progress reports to the directory that tensorboard is monitoring) :

```
python -m scripts.retrain \
  --bottleneck_dir=tf_files/bottlenecks \
  --how_many_training_steps=500 \
  --model_dir=tf_files/models/ \
  --summaries_dir=tf_files/training_summaries/"${ARCHITECTURE}" \
  --output_graph=tf_files/retrained_graph.pb \
  --output_labels=tf_files/retrained_labels.txt \
  --architecture="${ARCHITECTURE}" \
  --image_dir=tf_files/YOUR_DATASET_NAME
```
This script downloads the pre-trained model, adds a new final layer, and trains that layer on the dataset photos you've downloaded.
<br/><br/>
The above retraining command iterates only 500 times. You can very likely get improved results (i.e. higher accuracy) by training for longer. To get this improvement, remove the parameter --how_many_training_steps to use the default 4,000 iterations.
 
#### 4. Classifying an Image:
The repo also contains a copy of tensorflow's [label_image.py](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/image_retraining/label_image.py) example, which you can use to test your network. Take a minute to read the help for this script:

```
python -m  scripts.label_image -h
```
Now, let's run the script on a test image :
```
python -m scripts.label_image \
    --graph=tf_files/retrained_graph.pb  \
    --image=tf_files/YOUR_DATASET/FOLDER1/IMAGE_NAME
```
Each execution will print a list of  labels, in most cases with the correct on top (though each retrained model may be slightly different).

You might get results like this:
```
CLASS_OF_IMAGE (score = 0.99071)
OTHER_CLASS (score = 0.00595)
OTHER_CLASS (score = 0.00252)
OTHER_CLASS (score = 0.00049)
OTHER_CLASS (score = 0.00032)
```
This indicates a high confidence (~99%) that the image is a correctly classified image, and low confidence for any other label.
<br/><br/>
You can use label_image.py to classify any image file you choose, either from your downloaded collection, or new ones. You just have to change the --image file name argument to the script.
