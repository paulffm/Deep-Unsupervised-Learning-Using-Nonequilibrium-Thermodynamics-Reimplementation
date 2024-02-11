# Dirichlet Diffusion Score Model for Biological Sequence 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/paulffm/Deep-Unsupervised-Learning-Using-Nonequilibrium-Thermodynamics-Reimplementation-of-/blob/main/LICENSE)

Unofficial **PyTorch** reimplementation of the
paper [Deep Unsupervised Learning Using Nonequilibrium Thermodynamics](https://arxiv.org/pdf/1503.03585.pdf)
by Sohl-Dickstein et al.

<p align="center">
  <img src="sohl.png"  alt="1" width = 512px height = 173px >
</p>

Figure taken from [paper](https://arxiv.org/pdf/1503.03585.pdf).

## Usage

This implementation provides a example notebook for training and evaluation of this diffusion models for swiss roll dataset. As neural network, we implemented a simple MLP. The main training and evaluation procedure can be seen in the following:

```python
T = 50 # time steps
n_training_data = 100000 # number of samples to train on
n_samples = 5000 # number of samples

device = 'cuca'
mlp_model = MLP(hidden_dim=128).to(device) #nn

betas = torch.sigmoid(torch.linspace(-18, 10, T)) * (3e-1 - 1e-5) + 1e-5 # noise schedule
model = InitialDiffusionModel(T=40, betas=betas, model=mlp_model)
optimizer = torch.optim.Adam(mlp_model.parameters(), lr=1e-4)

# data
x0 = utils.sample_batch(n_training_data)
plt.scatter(x0[:, 0], x0[:, 1])
plt.show()

# train
model, loss = utils.train_loop(diffusion_model=model, optimizer=optimizer, batch_size=64000, nb_epochs=1000, device='cpu')
# sample
samples = model.sample(n_samples) 


```

## Note
This repository is a really simple implementation of Deep Unsupervised Learning Using Nonequilibrium Thermodynamics. However, it provides the main functionality and can be easily extended by adding other networks and dataset.
## Reference

```bibtex
@inproceedings{sohl-dicksteinDeepUnsupervisedLearning2015,
  title = {Deep {{Unsupervised Learning}} Using {{Nonequilibrium Thermodynamics}}},
  booktitle = {Proceedings of the 32nd {{International Conference}} on {{Machine Learning}}},
  author = {{Sohl-Dickstein}, Jascha and Weiss, Eric and Maheswaranathan, Niru and Ganguli, Surya},
  year = {2015},
  month = jun,
  pages = {2256--2265},
  publisher = {{PMLR}},
  issn = {1938-7228},
  urldate = {2023-01-07},
  langid = {english},
}
```
