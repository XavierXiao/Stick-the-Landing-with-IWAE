# Stick-the-Landing-with-IWAE

This is a pytorch implementation of Stick-the-Landing[1] by Roeder et. al. It is a method that reduce the variance estimation of reparametrization in VAE. The author also discussed incorporate this into IWAE, and here I also included IWAE in experiment. The inplementation of Stick-the-Landing is almost trivial, only with a stop_gradient operation after obtaining mu and log_sigma for the laten variables.

The experiment is run on MNIST. The network structure follows [2] with one stochastic layer. Batch size is 100, and optimizer is AdaDelta. Training is run 50 epoches. We compare the method for k=1 (where VAE and IWAE are equivalent) and k=5. The method is evaluated by computing NLL on test test. Results show that with this variance reduction technique improve the performance.

k=1:

|with variance reduction   | without variance reduction |
| ------------- | ------------- |
|92.02  | 92.67  |

k=5:

|VAE with variance reduction   | VAE without variance reduction |IWAE with variance reduction   | IWAE without variance reduction |
| ------------- | ------------- |------------- | ------------- |
| 91.62 | 91.96 |  89.98| 90.4  |


Reference:

[1] Roeder, Geoffrey, Yuhuai Wu, and David K. Duvenaud. "Sticking the landing: Simple, lower-variance gradient estimators for variational inference." Advances in Neural Information Processing Systems. 2017.

[2]Burda, Yuri, Roger Grosse, and Ruslan Salakhutdinov. "Importance weighted autoencoders." arXiv preprint arXiv:1509.00519 (2015).
