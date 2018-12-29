import matplotlib.pyplot as plt
import numpy as np
import re

# Plot loss from saved cluster output
f = open('output.txt', 'r')
losses = [float(num) for num in re.findall(r'Batch Loss: (.*)', f.read())]
iterations = [1 + 50*i for i in range(len(losses))]

# Plot loss and accuracy curves
plt.plot(iterations, losses, label='Cosine Mean Angle Error')
plt.yticks(np.arange(-1.0, max(losses), 0.1))
plt.legend()
plt.xlabel("Iterations")
plt.ylabel("Training Batch Loss")
plt.savefig('figures/loss_plot.png')
plt.close()