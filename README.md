# DLOPs Assignment 1 Report

**Name:** Laksh Mendpara  
**Roll Number:** B23CS1037  
**Colab Link:** [Enter Colab Link Here]

---

## Q1(a): ResNet Classification on MNIST and FashionMNIST

In this section, we trained ResNet-18 and ResNet-50 models on the MNIST and FashionMNIST datasets using various hyperparameters (Batch Size, Optimizer, Learning Rate).

### Results Analysis

- **Overall Performance:** The models performed exceptionally well on MNIST, achieving >99% accuracy in most configurations.
- **FashionMNIST Challenge:** FashionMNIST proved more challenging, with accuracies hovering around 90-93%.
- **Optimizer Comparison:** Adam generally converged faster and achieved better stability compared to SGD, especially at lower learning rates. SGD with LR=0.0001 showed significantly poorer performance on FashionMNIST.
- **Model Depth:** Interestingly, ResNet-18 often performed comparably or even slightly better than ResNet-50 on FashionMNIST within the 5-epoch limit. This suggests that for these relatively simple datasets, the deeper architecture of ResNet-50 might not grant a significant advantage without more extensive training or data augmentation, and might be more prone to overfitting or slower convergence.

### Experimental Results

#### MNIST Results

| Batch Size | Optimizer | Learning Rate | ResNet-50 (5e) | ResNet-50 (10e) | ResNet-18 (5e) | ResNet-18 (10e) |
| ---------- | --------- | ------------- | -------------- | --------------- | -------------- | --------------- |
| 16         | sgd       | 0.001         | 99.31          | 99.31           | 99.06          | 99.37           |
| 16         | sgd       | 0.0001        | 98.74          | 98.56           | 97.97          | 98.75           |
| 16         | adam      | 0.001         | 99.11          | 99.02           | 99.07          | 99.12           |
| 16         | adam      | 0.0001        | 99.17          | 99.15           | 99.15          | 99.42           |
| 32         | sgd       | 0.001         | 99.27          | 99.24           | 99.01          | 99.28           |
| 32         | sgd       | 0.0001        | 95.89          | 96.20           | 96.41          | 97.76           |
| 32         | adam      | 0.001         | 99.14          | 99.05           | 98.82          | 99.33           |
| 32         | adam      | 0.0001        | 99.31          | 99.10           | 99.25          | 99.41           |

#### FASHION-MNIST Results

| Batch Size | Optimizer | Learning Rate | ResNet-50 (5e) | ResNet-50 (10e) | ResNet-18 (5e) | ResNet-18 (10e) |
| ---------- | --------- | ------------- | -------------- | --------------- | -------------- | --------------- |
| 16         | sgd       | 0.001         | 90.54          | 91.74           | 89.81          | 90.90           |
| 16         | sgd       | 0.0001        | 83.21          | 84.82           | 84.97          | 86.63           |
| 16         | adam      | 0.001         | 88.07          | 89.85           | 90.79          | 92.99           |
| 16         | adam      | 0.0001        | 91.33          | 91.76           | 92.00          | 92.43           |
| 32         | sgd       | 0.001         | 88.94          | 90.13           | 89.02          | 90.52           |
| 32         | sgd       | 0.0001        | 79.52          | 80.64           | 81.65          | 82.31           |
| 32         | adam      | 0.001         | 89.65          | 91.09           | 90.46          | 92.00           |
| 32         | adam      | 0.0001        | 91.46          | 91.91           | 90.27          | 90.66           |

_Note: (5e) denotes Accuracy at Epoch 5, (10e) denotes Accuracy at Epoch 10._

### Performance Visualization

#### Parameter Grid Analysis

![Parameter Grid](q1_a_experiments/plots/full_parameter_grid.png)

#### Accuracy Matrix

![Accuracy Matrix](q1_a_experiments/plots/accuracy_matrix.png)

#### Overfitting Analysis

![Overfitting Analysis](q1_a_experiments/plots/overfitting_analysis.png)

---

## Q1(b): SVM Classification

We trained SVM classifiers with Polynomial and RBF kernels on both datasets.

### Results Analysis

- **Kernel Performance:** RBF kernels generally outperformed Polynomial kernels, offering a good balance between training speed and accuracy.
- **Effect of Degree:** For the Polynomial kernel, degree 2 performed significantly better than degree 3, which notably failed to converge/generalize well with C=0.1 on MNIST (accuracy < 50%).
- **Training Time:** SVM training was extremely efficient, taking only a few seconds for these datasets.

#### MNIST SVM Results

| Kernel | Params                                    | Test Acc (%) | Train Time (ms) |
| ------ | ----------------------------------------- | ------------ | --------------- |
| poly   | {'degree': 2, 'C': 0.1, 'gamma': 'scale'} | 87.37        | 4686            |
| poly   | {'degree': 2, 'C': 0.1, 'gamma': 'auto'}  | 87.01        | 4235            |
| poly   | {'degree': 2, 'C': 1.0, 'gamma': 'scale'} | 95.65        | 3136            |
| poly   | {'degree': 2, 'C': 1.0, 'gamma': 'auto'}  | 95.61        | 3081            |
| poly   | {'degree': 3, 'C': 0.1, 'gamma': 'scale'} | 45.98        | 5219            |
| poly   | {'degree': 3, 'C': 0.1, 'gamma': 'auto'}  | 44.33        | 5531            |
| poly   | {'degree': 3, 'C': 1.0, 'gamma': 'scale'} | 92.04        | 3809            |
| poly   | {'degree': 3, 'C': 1.0, 'gamma': 'auto'}  | 91.80        | 3872            |
| rbf    | {'C': 0.1, 'gamma': 'scale'}              | 91.15        | 3859            |
| rbf    | {'C': 0.1, 'gamma': 'auto'}               | 91.15        | 3788            |
| rbf    | {'C': 1.0, 'gamma': 'scale'}              | 96.01        | 3276            |
| rbf    | {'C': 1.0, 'gamma': 'auto'}               | 95.99        | 2991            |

#### FASHION-MNIST SVM Results

| Kernel | Params                                    | Test Acc (%) | Train Time (ms) |
| ------ | ----------------------------------------- | ------------ | --------------- |
| poly   | {'degree': 2, 'C': 0.1, 'gamma': 'scale'} | 75.16        | 3586            |
| poly   | {'degree': 2, 'C': 0.1, 'gamma': 'auto'}  | 75.16        | 3542            |
| poly   | {'degree': 2, 'C': 1.0, 'gamma': 'scale'} | 81.22        | 3055            |
| poly   | {'degree': 2, 'C': 1.0, 'gamma': 'auto'}  | 81.22        | 3015            |
| poly   | {'degree': 3, 'C': 0.1, 'gamma': 'scale'} | 71.80        | 3738            |
| poly   | {'degree': 3, 'C': 0.1, 'gamma': 'auto'}  | 71.80        | 3724            |
| poly   | {'degree': 3, 'C': 1.0, 'gamma': 'scale'} | 80.16        | 3085            |
| poly   | {'degree': 3, 'C': 1.0, 'gamma': 'auto'}  | 80.15        | 3058            |
| rbf    | {'C': 0.1, 'gamma': 'scale'}              | 77.23        | 3259            |
| rbf    | {'C': 0.1, 'gamma': 'auto'}               | 77.24        | 3183            |
| rbf    | {'C': 1.0, 'gamma': 'scale'}              | 82.44        | 3016            |
| rbf    | {'C': 1.0, 'gamma': 'auto'}               | 82.44        | 2847            |

### SVM Visualizations

#### Accuracy Trends

![SVM Trends](q1_b_experiments/plots/svm_accuracy_trends.png)

#### Polynomial Kernel Heatmap

![Poly Heatmap](q1_b_experiments/plots/svm_poly_heatmap.png)

#### RBF Kernel Heatmap

![RBF Heatmap](q1_b_experiments/plots/svm_rbf_heatmap.png)

---

## Q2: Hardware Comparison (CPU vs GPU)

This section compares the training performance of ResNet models on CPU and GPU.

### Results Analysis

- **Training Time:** using a GPU yields a massive speedup. Training ResNet-18 on GPU took ~300s (5 mins) compared to ~1700s (28 mins) on CPU, a speedup of roughly **5.5x**.
- **FLOPs:** As expected, ResNet-50 requires significantly more FLOPs (~4.1 GFLOPs) compared to ResNet-18 (~1.8 GFLOPs), which correlates with the increased training time.
- **Accuracy:** The final accuracy is comparable across devices, as expected, since the math is identical (ignoring minor non-determinism).

#### Performance Table

| Compute | Batch Size | Optimizer | LR    | ResNet-18 Acc | ResNet-32 Acc | ResNet-50 Acc | R18 Time(ms) | R32 Time(ms) | R50 Time(ms) | R18 FLOPs  | R32 FLOPs  | R50 FLOPs  |
| ------- | ---------- | --------- | ----- | ------------- | ------------- | ------------- | ------------ | ------------ | ------------ | ---------- | ---------- | ---------- |
| CPU     | 16         | sgd       | 0.001 | 90.48         | 90.83         | 88.52         | 1734187      | 2727171      | 4957158      | 1818558976 | 3670755840 | 4109485056 |
| CPU     | 16         | adam      | 0.001 | 90.53         | 90.46         | 89.20         | 2009823      | 2816809      | 4395270      | 1818558976 | 3670755840 | 4109485056 |
| GPU     | 16         | sgd       | 0.001 | 90.15         | 90.47         | 90.33         | 315674       | 464127       | 646635       | 1818558976 | 3670755840 | 4109485056 |
| GPU     | 16         | adam      | 0.001 | 90.79         | 91.02         | 88.07         | 326356       | 533276       | 635039       | 1818558976 | 3670755840 | 4109485056 |

_Note: Training times are in milliseconds._

### Hardware Performance Dashboard

![Dashboard](q2_experiments/plots/hardware_performance_dashboard.png)

![Return on Compute](q2_experiments/plots/return_on_compute.png)

---

## Appendix: Training Curves

### Q1(a) ResNet-18 Experiments (Max Epochs = 10)

#### MNIST

**ResNet-18, ADAM, LR=0.0001**
![Accuracy](q1_a_experiments/mnist_experiments/dataset_name_mnist_model_name_resnet18_learning_rate_0.0001_batch_size_16_optimizer_name_adam_num_epochs_10/accuracy_plot.png)
![Loss](q1_a_experiments/mnist_experiments/dataset_name_mnist_model_name_resnet18_learning_rate_0.0001_batch_size_16_optimizer_name_adam_num_epochs_10/loss_plot.png)
<br/>

**ResNet-18, SGD, LR=0.0001**
![Accuracy](q1_a_experiments/mnist_experiments/dataset_name_mnist_model_name_resnet18_learning_rate_0.0001_batch_size_16_optimizer_name_sgd_num_epochs_10/accuracy_plot.png)
![Loss](q1_a_experiments/mnist_experiments/dataset_name_mnist_model_name_resnet18_learning_rate_0.0001_batch_size_16_optimizer_name_sgd_num_epochs_10/loss_plot.png)
<br/>

**ResNet-18, ADAM, LR=0.0001**
![Accuracy](q1_a_experiments/mnist_experiments/dataset_name_mnist_model_name_resnet18_learning_rate_0.0001_batch_size_32_optimizer_name_adam_num_epochs_10/accuracy_plot.png)
![Loss](q1_a_experiments/mnist_experiments/dataset_name_mnist_model_name_resnet18_learning_rate_0.0001_batch_size_32_optimizer_name_adam_num_epochs_10/loss_plot.png)
<br/>

**ResNet-18, SGD, LR=0.0001**
![Accuracy](q1_a_experiments/mnist_experiments/dataset_name_mnist_model_name_resnet18_learning_rate_0.0001_batch_size_32_optimizer_name_sgd_num_epochs_10/accuracy_plot.png)
![Loss](q1_a_experiments/mnist_experiments/dataset_name_mnist_model_name_resnet18_learning_rate_0.0001_batch_size_32_optimizer_name_sgd_num_epochs_10/loss_plot.png)
<br/>

**ResNet-18, ADAM, LR=0.001**
![Accuracy](q1_a_experiments/mnist_experiments/dataset_name_mnist_model_name_resnet18_learning_rate_0.001_batch_size_16_optimizer_name_adam_num_epochs_10/accuracy_plot.png)
![Loss](q1_a_experiments/mnist_experiments/dataset_name_mnist_model_name_resnet18_learning_rate_0.001_batch_size_16_optimizer_name_adam_num_epochs_10/loss_plot.png)
<br/>

**ResNet-18, SGD, LR=0.001**
![Accuracy](q1_a_experiments/mnist_experiments/dataset_name_mnist_model_name_resnet18_learning_rate_0.001_batch_size_16_optimizer_name_sgd_num_epochs_10/accuracy_plot.png)
![Loss](q1_a_experiments/mnist_experiments/dataset_name_mnist_model_name_resnet18_learning_rate_0.001_batch_size_16_optimizer_name_sgd_num_epochs_10/loss_plot.png)
<br/>

**ResNet-18, ADAM, LR=0.001**
![Accuracy](q1_a_experiments/mnist_experiments/dataset_name_mnist_model_name_resnet18_learning_rate_0.001_batch_size_32_optimizer_name_adam_num_epochs_10/accuracy_plot.png)
![Loss](q1_a_experiments/mnist_experiments/dataset_name_mnist_model_name_resnet18_learning_rate_0.001_batch_size_32_optimizer_name_adam_num_epochs_10/loss_plot.png)
<br/>

**ResNet-18, SGD, LR=0.001**
![Accuracy](q1_a_experiments/mnist_experiments/dataset_name_mnist_model_name_resnet18_learning_rate_0.001_batch_size_32_optimizer_name_sgd_num_epochs_10/accuracy_plot.png)
![Loss](q1_a_experiments/mnist_experiments/dataset_name_mnist_model_name_resnet18_learning_rate_0.001_batch_size_32_optimizer_name_sgd_num_epochs_10/loss_plot.png)
<br/>

#### FASHION-MNIST

**ResNet-18, ADAM, LR=0.0001**
![Accuracy](q1_a_experiments/fashion_mnist_experiments/dataset_name_fashion-mnist_model_name_resnet18_learning_rate_0.0001_batch_size_16_optimizer_name_adam_num_epochs_10/accuracy_plot.png)
![Loss](q1_a_experiments/fashion_mnist_experiments/dataset_name_fashion-mnist_model_name_resnet18_learning_rate_0.0001_batch_size_16_optimizer_name_adam_num_epochs_10/loss_plot.png)
<br/>

**ResNet-18, SGD, LR=0.0001**
![Accuracy](q1_a_experiments/fashion_mnist_experiments/dataset_name_fashion-mnist_model_name_resnet18_learning_rate_0.0001_batch_size_16_optimizer_name_sgd_num_epochs_10/accuracy_plot.png)
![Loss](q1_a_experiments/fashion_mnist_experiments/dataset_name_fashion-mnist_model_name_resnet18_learning_rate_0.0001_batch_size_16_optimizer_name_sgd_num_epochs_10/loss_plot.png)
<br/>

**ResNet-18, ADAM, LR=0.0001**
![Accuracy](q1_a_experiments/fashion_mnist_experiments/dataset_name_fashion-mnist_model_name_resnet18_learning_rate_0.0001_batch_size_32_optimizer_name_adam_num_epochs_10/accuracy_plot.png)
![Loss](q1_a_experiments/fashion_mnist_experiments/dataset_name_fashion-mnist_model_name_resnet18_learning_rate_0.0001_batch_size_32_optimizer_name_adam_num_epochs_10/loss_plot.png)
<br/>

**ResNet-18, SGD, LR=0.0001**
![Accuracy](q1_a_experiments/fashion_mnist_experiments/dataset_name_fashion-mnist_model_name_resnet18_learning_rate_0.0001_batch_size_32_optimizer_name_sgd_num_epochs_10/accuracy_plot.png)
![Loss](q1_a_experiments/fashion_mnist_experiments/dataset_name_fashion-mnist_model_name_resnet18_learning_rate_0.0001_batch_size_32_optimizer_name_sgd_num_epochs_10/loss_plot.png)
<br/>

**ResNet-18, ADAM, LR=0.001**
![Accuracy](q1_a_experiments/fashion_mnist_experiments/dataset_name_fashion-mnist_model_name_resnet18_learning_rate_0.001_batch_size_16_optimizer_name_adam_num_epochs_10/accuracy_plot.png)
![Loss](q1_a_experiments/fashion_mnist_experiments/dataset_name_fashion-mnist_model_name_resnet18_learning_rate_0.001_batch_size_16_optimizer_name_adam_num_epochs_10/loss_plot.png)
<br/>

**ResNet-18, SGD, LR=0.001**
![Accuracy](q1_a_experiments/fashion_mnist_experiments/dataset_name_fashion-mnist_model_name_resnet18_learning_rate_0.001_batch_size_16_optimizer_name_sgd_num_epochs_10/accuracy_plot.png)
![Loss](q1_a_experiments/fashion_mnist_experiments/dataset_name_fashion-mnist_model_name_resnet18_learning_rate_0.001_batch_size_16_optimizer_name_sgd_num_epochs_10/loss_plot.png)
<br/>

**ResNet-18, ADAM, LR=0.001**
![Accuracy](q1_a_experiments/fashion_mnist_experiments/dataset_name_fashion-mnist_model_name_resnet18_learning_rate_0.001_batch_size_32_optimizer_name_adam_num_epochs_10/accuracy_plot.png)
![Loss](q1_a_experiments/fashion_mnist_experiments/dataset_name_fashion-mnist_model_name_resnet18_learning_rate_0.001_batch_size_32_optimizer_name_adam_num_epochs_10/loss_plot.png)
<br/>

**ResNet-18, SGD, LR=0.001**
![Accuracy](q1_a_experiments/fashion_mnist_experiments/dataset_name_fashion-mnist_model_name_resnet18_learning_rate_0.001_batch_size_32_optimizer_name_sgd_num_epochs_10/accuracy_plot.png)
![Loss](q1_a_experiments/fashion_mnist_experiments/dataset_name_fashion-mnist_model_name_resnet18_learning_rate_0.001_batch_size_32_optimizer_name_sgd_num_epochs_10/loss_plot.png)
<br/>

### Q2 Experiments (FashionMNIST, Max Epochs = 5)

#### CPU Experiments

**Resnet18 on CPU**
![Accuracy](q2_experiments/cpu_experiments/dataset_name_fashion-mnist_model_name_resnet18_learning_rate_0.001_batch_size_16_optimizer_name_adam_num_epochs_5_device_cpu/accuracy_plot.png)
![Loss](q2_experiments/cpu_experiments/dataset_name_fashion-mnist_model_name_resnet18_learning_rate_0.001_batch_size_16_optimizer_name_adam_num_epochs_5_device_cpu/loss_plot.png)
<br/>

**Resnet18 on CPU**
![Accuracy](q2_experiments/cpu_experiments/dataset_name_fashion-mnist_model_name_resnet18_learning_rate_0.001_batch_size_16_optimizer_name_sgd_num_epochs_5_device_cpu/accuracy_plot.png)
![Loss](q2_experiments/cpu_experiments/dataset_name_fashion-mnist_model_name_resnet18_learning_rate_0.001_batch_size_16_optimizer_name_sgd_num_epochs_5_device_cpu/loss_plot.png)
<br/>

**Resnet32 on CPU**
![Accuracy](q2_experiments/cpu_experiments/dataset_name_fashion-mnist_model_name_resnet32_learning_rate_0.001_batch_size_16_optimizer_name_adam_num_epochs_5_device_cpu/accuracy_plot.png)
![Loss](q2_experiments/cpu_experiments/dataset_name_fashion-mnist_model_name_resnet32_learning_rate_0.001_batch_size_16_optimizer_name_adam_num_epochs_5_device_cpu/loss_plot.png)
<br/>

**Resnet32 on CPU**
![Accuracy](q2_experiments/cpu_experiments/dataset_name_fashion-mnist_model_name_resnet32_learning_rate_0.001_batch_size_16_optimizer_name_sgd_num_epochs_5_device_cpu/accuracy_plot.png)
![Loss](q2_experiments/cpu_experiments/dataset_name_fashion-mnist_model_name_resnet32_learning_rate_0.001_batch_size_16_optimizer_name_sgd_num_epochs_5_device_cpu/loss_plot.png)
<br/>

**Resnet50 on CPU**
![Accuracy](q2_experiments/cpu_experiments/dataset_name_fashion-mnist_model_name_resnet50_learning_rate_0.001_batch_size_16_optimizer_name_adam_num_epochs_5_device_cpu/accuracy_plot.png)
![Loss](q2_experiments/cpu_experiments/dataset_name_fashion-mnist_model_name_resnet50_learning_rate_0.001_batch_size_16_optimizer_name_adam_num_epochs_5_device_cpu/loss_plot.png)
<br/>

**Resnet50 on CPU**
![Accuracy](q2_experiments/cpu_experiments/dataset_name_fashion-mnist_model_name_resnet50_learning_rate_0.001_batch_size_16_optimizer_name_sgd_num_epochs_5_device_cpu/accuracy_plot.png)
![Loss](q2_experiments/cpu_experiments/dataset_name_fashion-mnist_model_name_resnet50_learning_rate_0.001_batch_size_16_optimizer_name_sgd_num_epochs_5_device_cpu/loss_plot.png)
<br/>

#### GPU Experiments

**Resnet18 on GPU**
![Accuracy](q2_experiments/cuda_experiments/dataset_name_fashion-mnist_model_name_resnet18_learning_rate_0.001_batch_size_16_optimizer_name_adam_num_epochs_5_device_cuda/accuracy_plot.png)
![Loss](q2_experiments/cuda_experiments/dataset_name_fashion-mnist_model_name_resnet18_learning_rate_0.001_batch_size_16_optimizer_name_adam_num_epochs_5_device_cuda/loss_plot.png)
<br/>

**Resnet18 on GPU**
![Accuracy](q2_experiments/cuda_experiments/dataset_name_fashion-mnist_model_name_resnet18_learning_rate_0.001_batch_size_16_optimizer_name_sgd_num_epochs_5_device_cuda/accuracy_plot.png)
![Loss](q2_experiments/cuda_experiments/dataset_name_fashion-mnist_model_name_resnet18_learning_rate_0.001_batch_size_16_optimizer_name_sgd_num_epochs_5_device_cuda/loss_plot.png)
<br/>

**Resnet32 on GPU**
![Accuracy](q2_experiments/cuda_experiments/dataset_name_fashion-mnist_model_name_resnet32_learning_rate_0.001_batch_size_16_optimizer_name_adam_num_epochs_5_device_cuda/accuracy_plot.png)
![Loss](q2_experiments/cuda_experiments/dataset_name_fashion-mnist_model_name_resnet32_learning_rate_0.001_batch_size_16_optimizer_name_adam_num_epochs_5_device_cuda/loss_plot.png)
<br/>

**Resnet32 on GPU**
![Accuracy](q2_experiments/cuda_experiments/dataset_name_fashion-mnist_model_name_resnet32_learning_rate_0.001_batch_size_16_optimizer_name_sgd_num_epochs_5_device_cuda/accuracy_plot.png)
![Loss](q2_experiments/cuda_experiments/dataset_name_fashion-mnist_model_name_resnet32_learning_rate_0.001_batch_size_16_optimizer_name_sgd_num_epochs_5_device_cuda/loss_plot.png)
<br/>

**Resnet50 on GPU**
![Accuracy](q2_experiments/cuda_experiments/dataset_name_fashion-mnist_model_name_resnet50_learning_rate_0.001_batch_size_16_optimizer_name_adam_num_epochs_5_device_cuda/accuracy_plot.png)
![Loss](q2_experiments/cuda_experiments/dataset_name_fashion-mnist_model_name_resnet50_learning_rate_0.001_batch_size_16_optimizer_name_adam_num_epochs_5_device_cuda/loss_plot.png)
<br/>

**Resnet50 on GPU**
![Accuracy](q2_experiments/cuda_experiments/dataset_name_fashion-mnist_model_name_resnet50_learning_rate_0.001_batch_size_16_optimizer_name_sgd_num_epochs_5_device_cuda/accuracy_plot.png)
![Loss](q2_experiments/cuda_experiments/dataset_name_fashion-mnist_model_name_resnet50_learning_rate_0.001_batch_size_16_optimizer_name_sgd_num_epochs_5_device_cuda/loss_plot.png)
<br/>
