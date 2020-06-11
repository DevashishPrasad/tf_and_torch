# Pytorch_Scripts

This repository contains some random pytorch scripts for
<ul>
  <li>Building custom Models</li>
  <li>Data loading and visualizing</li>
  <li>Model Training and Optimizing</li>
  <li>Visualizing Model feature maps and kernels</li>
</ul>
And much more to come...

## Custom CNN Models made from scratch : 

1. **NN using ModuleList** <br>
    A simple Neural Network with nn.ModuleList(). You can define its architecture at the runtime.
    <ol>
    <li>Models/NN.py</li>
    </ol>

2. **VGG based models** <br>
    VGG is the simplest symmetric CNN to start with.
    <ol>
    <li> Models/VGG_6.py </li>
    <li> Models/VGG_16.py </li>
    </ol>

3. **ResNet based models**  <br>
    ResNet (Residual Networks) is another simple CNN which uses skip connections to solve vanishing gradient problems and go even deeper.
    <ol>
    <li> Models/ResNet_8.py </li>
    <li> Models/ResNet_12.py </li>
    <li> Models/Resnet_18.py	<a href="Model graphs/resnet18.onnx.png">(graph)</a></li>
    </ol>

## Siamese of Triplet nets made from scratch : 

1. **Siamese Network**  <br>
    Siamese networks are the twin networks that try to learn the best possible embeddings for the input data. We then use distance between these embeddings to distinguish between images.
    <ol>
    <li> Models/Simple_siamese.py </li>
    </ol>

2. **Triplet Network**  <br>
    Same as the siamese networks but here there are thress netwokrs that try to learn the possible embeddings for the input data.
    <ol>
    <li> Models/Simple_triplet.py </li>
    </ol>

I use Mnist dataset as an example for both of the above models. 

## Training
The training folder contains various training pipelines.

## Visualize Model
In this folder you can find scripts to visualize the kernels and weights learnt by the model. It is a good way to audit the learning process of the models.

## And BTW Pytoch is Amazing !! ;-)
