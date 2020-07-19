# Neuromorphic-Digit-Classification
 **Simple Python simulation of a one-hidden-layer spiking neural network for digit classification on sklearn's "digits" dataset.**

The topology of the network is shown in the figure below. The input image is first **flatten and rescaled**. The resulting vector is then **fully-connected to 256 LIF neurons** (with bias current source as well), the weights for each neuron are **randomly sampled from the unit hypersphere**. The spike trains of each neuron are then filtered using **PSC kernels**. The time-dependent output vector "p(t)" is then **decoded using a linear decoder "A" learned via ridge-regression**. The result is a **one-hot encoded vector "d(t)"** representing the class of the input image.

![Alt text](visuals/Network_topo.png?raw=true "SNN topology")

# Accuracy of the SNN in function of time

Digit images are shown to the network at the **0.025 second time instant**, the inference accuracy of the network on the test set rises to **90%** in about **10 milliseconds**.

![Alt text](visuals/Accuracy_evo.png?raw=true "Accuracy of the model in function of time")

# Network activity

The following raster plot shows the spiking activity (256 neurons) of the network for an **input image representing the digit "0"**. The image is shown to the network at the **0.025 second time instant** therefore there is **no activity** before that instant. Even when the image is shown, the **spatio-temporal activity of the network is sparse**. Lack of any activity when no image is shown and sparsity of the network activity when an image is shown illustrate the fact that **event-based neuromorphic architectures are well-suited for ultra-low energy consumption figures when running on dedicated hardware**.

![Alt text](visuals/neural_act.png?raw=true "Network activity")

# How to cite

May you use this work, please cite:

```
@software{ali_safa_2020_3951560,
  author       = {Ali Safa},
  title        = {{Neuromorphic-Digit-Classification: Digit 
                   Classification Using a One-Hidden-Layer Spiking
                   Neural Network}},
  month        = jul,
  year         = 2020,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.3951560},
  url          = {https://doi.org/10.5281/zenodo.3951560}
}
```
