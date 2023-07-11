import random
import matplotlib.pyplot as plt

file_path = "/data/home/hemantmishra/examples/CrypTen/output_jj.txt"  # Replace with the actual file path

loss_values = []

with open(file_path, 'r') as file:
    lines = file.readlines()
epochs = 0
for i in range(0, len(lines), 6):
    epochs += 1
    current_loss = float(lines[i+2].split(": ")[1])
    loss_values.append(current_loss)


plt.plot(range(epochs), loss_values, label='Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Values over Epochs')
plt.legend()
plt.show()
