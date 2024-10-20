import numpy as np
import yaml
import sympy as sp
import emcee
import corner
import matplotlib.pyplot as plt
from sympy import symbols, sin, log

## --------------------------------------------------------------------------------------------------------------------------------------- ##

# function that creates a synthetic dataset
def getSyntheticDataset(config):

    print("## Generation of a synthetic dataset..")
    xmin = config['synthetic_dataset']['xmin']
    xmax = config['synthetic_dataset']['xmax']
    npoints = config['synthetic_dataset']['num_points']
    x_dist = config['synthetic_dataset']['x_distribution']
    func_data_str = config['synthetic_dataset']['generative_function']
    print("## Using ", func_data_str, " as generative function..")
    data_params = config['dataset_parameters']
    param_values = [data_params[key] for key in sorted(data_params.keys())]
    data_func = create_function_from_string(func_data_str, data_params)
    noise_level = config['synthetic_dataset']['noise_level']
    yerr = config['synthetic_dataset']['default_yerr']

    if x_dist == 'linear' or x_dist == '':
        x = np.linspace(xmin, xmax, npoints)
    elif x_dist == 'log':
        x = np.logspace(xmin, xmax, npoints)

    y = data_func(x, *param_values)
    yerr = np.ones_like(y) * yerr

    with open("./dataset.txt", "w") as file:
        for ii in range(0,len(x)):
            file.write(str(x[ii])+" "+str(y[ii])+" "+str(yerr[ii])+"\n")

    print("## Written synthetic dataset at ./dataset.txt")

    return x, y, yerr

## --------------------------------------------------------------------------------------------------------------------------------------- ##

# function that translates a string into a python function (used for custom models)
def create_function_from_string(func_str, params):
    x = sp.symbols('x')
    param_symbols = {param: sp.symbols(param) for param in params}
    func = eval(func_str, {"x": x, "sin": sp.sin, "log": sp.log, **param_symbols})

    def fit_func(x_val, *param_values):
        param_dict = dict(zip(param_symbols.values(), param_values))
        return sp.lambdify(x, func.subs(param_dict), 'numpy')(x_val)

    return fit_func

## --------------------------------------------------------------------------------------------------------------------------------------- ##

# Log-likelihood function used in MCMC fitting
def log_likelihood(theta, x, y, func, yerr):
    model = func(x, *theta)
    sigma2 = yerr ** 2
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

## --------------------------------------------------------------------------------------------------------------------------------------- ##

# log-prior function used in MCMC fitting
def log_prior(theta, prior_ranges):
    for i, (low, high) in enumerate(prior_ranges):
        if not (low < theta[i] < high):
            return -np.inf
    return 0.0

## --------------------------------------------------------------------------------------------------------------------------------------- ##

# log posterior = log prior + log likelihood (used in MCMC fitting)
def log_posterior(theta, x, y, func, yerr, prior_ranges):
    lp = log_prior(theta, prior_ranges)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, func, yerr)

## --------------------------------------------------------------------------------------------------------------------------------------- ##

# Esegui MCMC per campionare i parametri
def run_mcmc(x, y, func, initial_params, yerr, prior_ranges, nwalkers, steps):
    ndim = len(initial_params)
    pos = initial_params + 1e-4 * np.random.randn(nwalkers, ndim)  # Inizializzazione random
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(x, y, func, yerr, prior_ranges))
    sampler.run_mcmc(pos, steps, progress=True)
    return sampler

## --------------------------------------------------------------------------------------------------------------------------------------- ##

# Genera un contour plot dei risultati con contorni a 1, 2 e 3 sigma
def plot_contours(sampler, param_names, discard):
    samples = sampler.get_chain(discard=discard, flat=True)
    fig = corner.corner(samples, labels=param_names, plot_datapoints=False, 
                        plot_contours=True, fill_contours=False, levels=[0.68, 0.95, 0.997])
    plt.show()

## --------------------------------------------------------------------------------------------------------------------------------------- ##

# plot the MCMC chains
def plot_chains(sampler, param_names):
    fig, axes = plt.subplots(len(param_names), figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    for i in range(len(param_names)):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(param_names[i])
    axes[-1].set_xlabel("Step")
    plt.show()

## --------------------------------------------------------------------------------------------------------------------------------------- ##

# extract the best fit parameters from chains along with parameter uncertainties
def get_best_fit_params(sampler,discard):
    samples = sampler.get_chain(discard=discard, flat=True)
    best_fit = np.mean(samples, axis=0)
    param_errors = np.std(samples, axis=0)
    return best_fit, param_errors

## --------------------------------------------------------------------------------------------------------------------------------------- ##

# Simple plot
def plotDataset(x,y,yerr):

    plt.title("Considered dataset")
    plt.errorbar(x, y, yerr=yerr, fmt='o', capsize=5, color='gray')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

## --------------------------------------------------------------------------------------------------------------------------------------- ##

# function that plots the dataset and models from MCMC
def plot_dataset_and_fit(x, y, fit_func, best_fit_params, param_errors, param_names, yerr=None):

    nmodels = 50
    delta = np.abs(np.max(x)-np.min(x))
    x_fit = np.linspace(np.min(x)-0.1*delta, np.max(x)+0.1*delta, 200)
    y_fit = fit_func(x_fit, *best_fit_params)

    plt.plot(x_fit, y_fit, label="Best-fit model", color='firebrick', lw=2)

    for i in range(nmodels):
        # Estrai parametri campionati dalla distribuzione normale (best-fit ± errori)
        sampled_params = np.random.normal(best_fit_params, param_errors)
        # Calcola la funzione con i parametri campionati
        y_fit_sampled = fit_func(x_fit, *sampled_params)
        # Plotta il modello campionato
        plt.plot(x_fit, y_fit_sampled, color='orange', alpha=0.1, lw=1)

    plt.errorbar(x, y, yerr=yerr, fmt='o', label="Data", capsize=5, color='gray')
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(np.min(x)-0.1*delta, np.max(x)+0.1*delta)
    #plt.xscale('log')
    #plt.yscale('log')
    plt.legend()
    plt.show()

## --------------------------------------------------------------------------------------------------------------------------------------- ##

# main function
def main(config_file):

    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    # create a synthetic dataset or read it from an existing file
    if config['dataset']['path']:
        data = np.loadtxt(config['dataset']['path'])
        x, y = data[:, 0], data[:, 1]
        yerr = data[:, 2] if data.shape[1] > 2 else np.ones_like(y) * config['synthetic_dataset']['default_yerr']
    else:
        x, y, yerr = getSyntheticDataset(config)

    # show the dataset before fitting
    plotDataset(x,y,yerr)

    # define the fitting function
    if config['model']['type'] == 'custom':
        func_str = config['model']['function']
        params = config['model_parameters']
        fit_func = create_function_from_string(func_str, params)
    else:
        raise ValueError("## Only custom functions are implemented up to now..")

    # extract parameters from configuration file
    initial_params = list(config['model_parameters'].values())
    prior_ranges = config['prior_ranges']  # Assume che il file di configurazione contenga i range dei prior
    nsteps_chains = config['steps']
    burnin = config['burnin']
    nwalkers = config['nwalkers']

    # MCMC fitting
    sampler = run_mcmc(x, y, fit_func, initial_params, yerr, prior_ranges, nwalkers, nsteps_chains)

    # plot the chains
    plot_chains(sampler, list(config['model_parameters'].keys()))

    # plot the contours of fitted parameters
    plot_contours(sampler, list(config['model_parameters'].keys()), burnin)

    # get the best-fit parameters
    best_fit_params, param_errors = get_best_fit_params(sampler,burnin)
    print(f"Best-fit parameters: {best_fit_params}")
    print(f"Parameter uncertainties: {param_errors}")

    # Plot the best-fit model on top of dataset
    plot_dataset_and_fit(x, y, fit_func, best_fit_params, param_errors, list(config['model_parameters'].keys()), yerr)


## --------------------------------------------------------------------------------------------------------------------------------------- ##

if __name__ == "__main__":
    main("./config.yaml")
