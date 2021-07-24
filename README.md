Messing around with opencv in python

# Getting Started

We use a virtual environment to do all our development in, this ensures that we don't import modules that accidentally conflict with other project work. 

1. `python3 -m venv venv` - this will create your virtual environment inside a `venv` folder
2. `source venv/bin/activate` - this will activate your virtual env
3. `python3 -m pip install -r requirements.txt`

## Adding new modules

New modules should be added to the virtual env with the following command: `pip install <module-name>`
After the module has been added you should update the `requirements.txt` file using: `pip freeze > requirements.txt`


If VSC is not linting or suggestions aren't working correctly then: 

1. press cmd + shift + p to bring up command palette and search for python: select interpreter
2. Pick then one which is associated with your virtual environment


# Part 1. Basic image manipulations:

- displaying (colour, grayscale,...)
- rotating
- resizing

code can be found in `src/001-basics-img-manipulations/basic_img_manipulations.py`

# Part 2. Plotting shapes

code can be found in `src/002-plotting-shapes/plotting_shapes.py`


# Part 3. Spatial Domain Image pre-processing

We use a variety of techniques to prepare images for our models to achieve better results. To better understand this it's useful to take a step back and try to understand how we perceive images. 

What image do you see below? 

Do you see a face looking directly at you or perhaps you have focused on the snozz and noticed the man also appears to be shown in a sideways perspective. 

![image trickery](https://miro.medium.com/max/800/1*ml9HFgcLeIqyGTb8HBJQYA.png)

What about this one? The artist has managed to capture a 3D like effect with only a piece of pen and paper. 

![cats](https://do.lolwot.com/wp-content/uploads/2015/05/20-mind-bending-optical-illusions-that-will-make-you-look-twice-9.jpg)

If we are to understand images we can see that edges and contour detection is pretty important for our perception. 

There's also other things to consider when thinking about images. It's remarkable how we are able to tell 2 objects apart from one another and even identify the target amongst other objects in the background. Even toddlers are capable of recognising objects at different scale, colour and rotation however this is not a trivial task for a computer that often has a hard time trying to identify pixels which come from the object we want to label and the one in the background. 

Fortunately there are some steps which can make edge detection easier. 

Spatial domain processing = "direct" processing

Examples:
- Intensities (Brightness, negative, gamma)
- Histogram (Stretching, equalisation)
- Spatial Filtering (Mean/median/gaussian/gradient-based filter, laplacian/unsharp masking)

### Intensities

Graylevels

![thumnail](assets/spatial_images/surfing_thumb.png)
![grayscale](assets/spatial_images/grayscale.png)
![negative](assets/spatial_images/negative.jpg)
![thresholded](assets/spatial_images/thresholded.png)

## Blurring

Goal: Reduce noise
Why?: Images with high resolution often have information in the background we aren't interested in. Blurring can help reduce noise. It also helps mimic real life conditions. Not every picture is shot perfectly and in perfect lighting. Using blurred images as part of our dataset will help make our models more robust.

Warning: Blurring too much can cause data to be lost!

![blur example](https://miro.medium.com/max/1400/1*Py75v-74yoCNA8PgLU6wEw.png)

4 main types of blur: 

- Averaging
- Gaussian
- Median
- Bilateral


# CNN introduction

Convolution - Mathematical operation on 2 objects to produce an outcome that expresses how the shape of one is modified by the other. Convolution is using a kernel to extract certain features from an input image. 

Feature map - Is the output activations of a given feature applied on an image. 

Kernel - A 2D matrix which is slid across an image and multiplied with the input

Examples of convolution:

![cnn](assets/convolution_in_action.gif)

![convolution](assets/convolution.png)

Why do we stack convolution layers?

It allows layers close to the input to learn low-level features (e.g. lines) and layers deeper into the model to learn high-order or more abstract features like shapes or specific objects. 

## Kernel vs filter

A filter is a concatenation of multiple kernels. Filters are always one dimensionality higher than the kernel and can be seen as multiple kernels stacked ontop of each other where every kernel is for a particular channel.
So in 2D convolutions, filters are 3D matrices. 

More info on different types of CNN [here](https://towardsdatascience.com/types-of-convolution-kernels-simplified-f040cb307c37).

![cnn diagram](assets/cnn_diagram.png)

## Padding and stride

Why?: Pixels at the corners of an image aren't calculated with the same amount of weighting as the center. If you apply convolution repeatedly you may lose this information.

Result: Gives addional pixels at the boundaries of the data. 


![padding](assets/padding.gif)

From the gif you can see that we move along one pixel at a time but we can change this number to be whatever we want. This is known as the stride. 

But why would we want to increase the stride? Surely that means we lose more information?

One of the limitations of the feature map is that it records precise positions of features in the input. Which means if the features of the image (e.g lines) were changed from some transformation (blurring, cropping, rotation etc.) then a different feature map would be produced. 

This is solved by a technique called down sampling which aims to capture the most important parts of the data without fine details. This is done by creating a lower resolution version of the input signal. Increasing the stride is one example of this but another popular method is called pooling.

![padding with zeroes](assets/cnn_with_padding.png)

## Pooling

A pooling layer is a new layer added after the convolutional layer. Specifically, after a nonlinearity (e.g. ReLU) has been applied to the feature maps output by a convolutional layer.

The addition of a pooling layer after the convolutional layer is a common pattern used for ordering layers within a convolutional neural network that may be repeated one or more times in a given model.

The pooling layer operates upon each feature map separately to create a new set of the same number of pooled feature maps.

Pooling involves selecting a pooling operation, much like a filter to be applied to feature maps. The size of the pooling operation or filter is smaller than the size of the feature map; specifically, it is almost always 2×2 pixels applied with a stride of 2 pixels.

This means that the pooling layer will always reduce the size of each feature map by a factor of 2, e.g. each dimension is halved, reducing the number of pixels or values in each feature map to one quarter the size. For example, a pooling layer applied to a feature map of 6×6 (36 pixels) will result in an output pooled feature map of 3×3 (9 pixels).

Average Pooling: Calculate the average value of each patch on the feature map. 
Max Pooling: Calculate the max value of each patch in the feature map. 

Pros:

- Improve translational invariance
- Downsampling: Reduce memory and computational cost
- Downsampling: Look more globally

![max pooling](assets/max-pooling.png)

The result of using a pooling layer and creating down sampled or pooled feature maps is a summarized version of the features detected in the input. They are useful as small changes in the location of the feature in the input detected by the convolutional layer will result in a pooled feature map with the feature in the same location. This capability added by pooling is called the model’s invariance to local translation.

"In all cases, pooling helps to make the representation become approximately invariant to small translations of the input. Invariance to translation means that if we translate the input by a small amount, the values of most of the pooled outputs do not change." - Deep Learning 2016

## Activation function

Theres a lot of different activation functions we can use e.g. sigmoid, tanh etc. but ReLU is the one that is typically used with CNN's.

![activation functions](assets/activation_fns.png)

Pros: 

- Faster convergence
- Easier and faster calculations
- Lower probability of vanishing gradient

Cons:

- Dying ReLU -> It can die if the inputs fall into the negative half (There are some other types of ReLU which can help solve this problem)

## Batch Normalization

Network is easier to train if the input is normalised (e.g. zero mean, unit variance). However in the CNN the paramaters are constantly changing which leads to data having a different distribution per layer. 

To solve this problem we introduce batch normalization. 


![batch normalization](assets/batch_norm.png)

## Fully connected layers

The purpose of the fully connected layers is to flattern the output of the convolution layers to create a single long feature vector. Each output is connected to ALL flatterned features (hence the name fully connected layers).

![fully connected layers](assets/fully-connected-layer.png)


## Part 4 Morphological Image processing

With Morphological image processing we are interested in the shape/form/structure of the image and their characteristics and relationships.

It has various use cases:

- Cleaning/smoothing images => i.e. removing excess data that may confuse our network
- Image analysis => detecting patterns/corners, boundary extraction

Structuring Elements (SE) are sub images (think of images as sets so these are sub sets.) used to probe images to find interesting properties. Similar in concept to filter kernels. 

An example makes this clearer to see:

![SE example](assets/004_morphological/structuring_elem_example.png)

SE specifies which neighbouring pixels are considered in determining the fate of the pixel in consideration.

### Hits and Fits

- Fit: *All* the on-pixels (coloured) in the SE fit the section on the image
- Hit: *Any* of the on-pixels in the SE fit the section on the image.

![Hits and fits](assets/004_morphological/hits-fits.png)

### Erosion

![Erosion formulae](assets/004_morphological/erosion-formula.png)

This can be thought as the erosion of image A by the SE B.

The SE B is positioned at a location in A and the new pixel value is determined to be: 

- 1 if B *fits* A
- 0 otherwise

Remember this is normally applied to binary images so 0 = Black and 1 = White.

![Erosion example](assets/004_morphological/erosion-example.png)

Use cases: 

- Can split two objects in the image
- Removes small inflections from an object. 

## Dilation

The dilation of image A by set B.

![Dilation Formula](assets/004_morphological/dilation-formula.png)

The SE B is positioned at a location in A and the new pixel value is determined to be: 

- 1 if B *hits* A
- 0 otherwise

![Dilation Example](assets/004_morphological/dilation-example.png)

Enlarges an image!

Use Cases: 

- Fills in gaps/breaks in an object. => This is useful in repairing text.

Things to note: 

- Erosion is not the inverse of dilution.  
- Dilution is not the inverse of Erosion.

### Opening

Opening = Erosion -> Dilation with same SE

![opening](assets/004_morphological/opening-formula.png)

Use cases: 
- Smoothes object boundaries
- Eliminates extrusions
- Can split object aparts
- Removes isolated pixels/pixel groups 

### Closing

Closing = Dilation -> Erosion with same SE

Use cases: 
- Smoothes object boundary
- Eliminates intrusions
- Can link closely object
- Removes holes

Opening is not the inverse of closing and vice versa. 

![Properties of Opening and closing](assets/004_morphological/properties-opening-closing.png)

![Fingerprint example](assets/004_morphological/fingerprint-example.png)

Tutorial example

![closing](assets/004_morphological/tut/closing.png) Closing
![opening](assets/004_morphological/tut/opening.png) Opening
![erosion](assets/004_morphological/tut/eroded.png) Erosion
![dilation](assets/004_morphological/tut/dilation.png) Dilation

### Boundary Extraction

Applications:
- Motion detection
- background/foreground segmentation (replacing the bg/fg of an image with another)

The contour-detection algorithms in OpenCV work very well, when the image has a dark background and a well-defined object-of-interest. 
But when the background of the input image is cluttered or has the same pixel intensity as the object-of-interest, the algorithms don’t fare so well.

![boundary extraction](assets/004_morphological/boundary-extraction.png)

### Filling in holes

Steps:
1. Read img
1. Threshold to binary img
1. Flood fill from pixel (0,0). Note how background has swapped from black -> white
1. Invert the image
1. Combine the thresholded image with the inverted flood-filled image using bitwise OR operation

![fill holes](https://learnopencv.com/wp-content/uploads/2015/11/imfill-opencv-steps.jpg)

Another way to fill images is by using a combination of techniques we have seen above: 

![region filling](assets/004_morphological/region-filling.png)