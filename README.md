# Neuromorphic-Digit-Classification
 Simple Python simulation of a one-hidden-layer spiking neural network for digit classification on sklearn's "digits" dataset.
 
**Output 1: Accuracy of the SNN in function of time**

Digit images are shown to the network at the **0.025 second time instant**, the inference accuracy of the network on the test set rises to **90%** in about **10 milliseconds**.

![Alt text](Accuracy_evo.png?raw=true "Accuracy of the model in function of time")

**Output 2: Network activity**

The following plot shows the spiking activity (256 neurons) of the network for an input image representing the digit **"0"**. The image is shown to the network at the **0.025 second time instant** therefore there is **no activity** before that instant. Even when the image is shown, the **spatio-temporal activity of the network is sparse** which illustrates well the fact that **event-based neuromorphic hardware are well-suited for ultra-low energy consumption figures**.

![Alt text](neural_act.png?raw=true "Network activity")
