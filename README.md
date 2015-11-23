# Fast Multi-threaded VGG 19 Feature Extractor

### Overview
This allows you to extract deep visual features from a pre-trained [VGG-19 net](http://arxiv.org/abs/1409.1556) for collections of images in the millions. Images are loaded and preprocessed in parallel using multiple CPU threads then shipped to the GPU in minibatches for the forward pass through the net. Model weights are downloaded for you and loaded using Torch's loadcaffe library, so you don't need to compile Caffe.

The feature extractor computes a 4096 dimensional feature vector for every image that contains the activations of the hidden layer immediately before the VGG's object classifier. The activations are ReLU-ed and L2-normalized, which means they be used as generic off-the-shelf features for tasks like classification or [image similarity](http://arxiv.org/abs/1505.07647).

### Example usage
You point it a tab separated file of (image_id, path to image on disk) e.g.
```
12      /home/username/images/12.jpg
342     /home/username/images/342.jpg
169     /home/username/images/169.jpg
```

specified by the ```-data``` flag, and it creates a tab separated file of (image_id, json encoded VGG vector) e.g.
```
12      [4096 dimensional vector]
342     [4096 dimensional vector]
169     [4096 dimensional vector]
```
specified by the ```-outFile``` flag.

```-nThreads``` tells it how many CPU loader threads to use. ```-batchSize``` tells it how many images to put in each minibatch. The higher the batchSize, the higher the throughput, so I'd make this as large as your GPU memory will allow.

Example:
```bash
th main.lua -data [tab separated file of (image_id, path_to_image_on_disk)] -outFile out_vecs -nThreads 8 -batchSize 128
```

### Requirements
- [Install torch on a machine with CUDA GPU](http://torch.ch/docs/getting-started.html#_)
- If on Mac OSX, run `brew install coreutils findutils` to get GNU versions of `wc`, `find`, and `cut`
- [loadcaffe](https://github.com/szagoruyko/loadcaffe)


### Why should I care about pre-trained deep convnet features in the first place?
- **They're powerful and transferable**: [Razavian et. al.](http://arxiv.org/pdf/1403.6382v3.pdf) show that these kinds of deep features can be used off-the-shelf to beat highly tuned state-of-the-art methods on challenging fine grained classification problems. That is, you can use the same features that distinguish a boat from a motorcycle to accurately tell two species of birds apart, even when the differences between species are extremely subtle. They show superior results to traditional feature representations (like SIFT, HOG, visual bag of words).
- **They're interpretable**: [Zeiler and Fergus](http://arxiv.org/abs/1311.2901) shows that the learned representations are far from a black box. They're actually quite interpretable: lower layers of the network learn filters that fire when they see color blobs, edges, lines, corners.

![](https://github.com/coreylynch/vgg-19-feature-extractor/blob/master/resources/MiddleLayers.png)

Middle layers see combinations of these lower level features, forming filters that respond to common textures. 

![](https://github.com/coreylynch/vgg-19-feature-extractor/blob/master/resources/LowerLayers.png)

Higher layers see combinations of these middle layers, forming filters that respond to object parts, and so on. 

![](https://github.com/coreylynch/vgg-19-feature-extractor/blob/master/resources/HigherLayers.png)

[source](https://courses.cs.washington.edu/courses/cse590v/14au/cse590v_dec5_DeepVis.pdf)

You can see the actual content of the image becoming increasingly explicit along the processing hierarchy.
- **They're cheap**: You only need to do one forward pass on a pre-trained net to get them.
- **They're the go-to visual component in some pretty incredible new machine vision applications**: like [automatically](http://cs.stanford.edu/people/karpathy/deepimagesent/) [describing](http://arxiv.org/abs/1411.4555) [images](http://arxiv.org/abs/1412.6632) from raw pixels. 

![](https://github.com/coreylynch/vgg-19-feature-extractor/blob/master/resources/pretrained.png)

Or being able to [embed images and words in a joint space](http://arxiv.org/abs/1411.2539) then do vector arithmetic in the learned space:

![](https://github.com/coreylynch/vgg-19-feature-extractor/blob/master/resources/multimodalEmbed.png)

Yep that's a multimodal vector describing a blue car minus the multimodal vector for the word "blue", plus the vector for "red" resulting in a vector that is [near images of red cars](https://media.giphy.com/media/EldfH1VJdbrwY/giphy.gif).

### When should I use these features?
Take advice from [here](http://cs231n.github.io/transfer-learning/#tf) (actually go read the entire course, it's amazing).

### Thanks to
Soumith Chintala for the scalable loader [starter code](https://github.com/soumith/imagenet-multiGPU.torch/blob/master/README.md), [Andrej Karpathy's course](http://cs231n.github.io) for teaching me about all this stuff in the first place .
