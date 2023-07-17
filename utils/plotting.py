import matplotlib.pyplot as plt

def plot_history(epochs, **kwargs):
    plt.figure()
    for key, value in kwargs.items():
        plt.plot(range(1, epochs+1), value, label=key)
    plt.title('History of ' + ', '.join(kwargs))
    plt.legend()
