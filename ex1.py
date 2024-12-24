import torch
import matplotlib.pyplot as plt
import botorch
from models import *

device = "cuda:0" if torch.cuda.is_available() else "cpu"

def objective(x):
    x = (x * 10) - 5
    return (1.4 - 3 * x) * torch.sin(1.5 * x) / 8

def plot_posterior(train_x, train_y, posterior, model_name):
    # plot true values
    plt.plot(test_x.cpu(), test_y.cpu(), color="black", linestyle="dashed", label="True Function", linewidth=2)
    plt.scatter(train_x.cpu(), train_y.cpu(), color="black")

    # plot mean and std
    mean = posterior.mean.detach().squeeze().cpu()
    std = torch.sqrt(posterior.variance).detach().squeeze().cpu()
    plt.plot(test_x.squeeze().cpu(), mean, color="C0", label="Mean", linewidth=2)
    plt.gca().fill_between(test_x.squeeze().cpu(), mean - 2 * std, mean + 2 * std, label=r'2$\sigma$', alpha = 0.2, color="orange")

    # plot function draws
    draws = posterior.rsample(torch.Size([8])).detach().cpu().squeeze()
    for i, draw in enumerate(draws):
        if i == 0: # add one legend
            plt.plot(test_x.squeeze().cpu(), draw, color="C3", alpha=0.3, label="Function Draw")
        else:
            plt.plot(test_x.squeeze().cpu(), draw, color="C3", alpha=0.3)

    plt.ylim(-1.8, 1.8)
    plt.legend(loc="lower center")
    plt.title("%s Uncertainty" % model_name)
    plt.show()

test_x = torch.linspace(0, 1, 100).double().unsqueeze(-1).to(device)
test_y = objective(test_x)

train_x = torch.tensor([[0.7576],
        [0.2793],
        [0.4031],
        [0.7347],
        [0.0993],
        [0.7999],
        [0.7544],
        [0.8695],
        [0.4388]]).double().to(device)
train_y = objective(train_x)

gp = SingleTaskGP(model_args={}, input_dim=1, output_dim=1)
gp.fit_and_save(train_x, train_y, None)
plot_posterior(train_x, train_y, gp.posterior(test_x), "GP")

# ibnn = SingleTaskIBNN({
#       "var_b": 1.3,
#       "var_w": 10,
#       "depth": 3
# }, 1, 1, device)
# ibnn.fit_and_save(train_x, train_y, None)
# plot_posterior(train_x, train_y, ibnn.posterior(test_x), "I-BNN")