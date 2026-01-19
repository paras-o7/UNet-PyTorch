import torch
from torch.utils.data import DataLoader
from semdronedataset import SemanticDroneDataset
from unet import UNet
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision.transforms.functional as TF

# HYPERPARAMETERS
EPOCHS = 125
BATCH_SIZE = 11
LEARNING_RATE = 5e-5
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.xpu.is_available():
    DEVICE = "xpu"
else:
    DEVICE = "cpu"
SHUFFLE = True
PIN_MEMORY = True
SAVE_DIR = "./model_save/binary.pth"
NUM_WORKERS = 16
PREFETCH_FACTOR = 16


plt.ion()
fig, ax = plt.subplots()
losses = []
line, = ax.plot([], [], linewidth=2)
ax.set_xlabel("Iteration")
ax.set_ylabel("Cost")
ax.set_title("Training Loss")
loss_text = ax.text(
    0.02, 0.95, "",
    transform=ax.transAxes,
    fontsize=10,
    verticalalignment="top"
)


dataset = SemanticDroneDataset("./archive/classes_dataset/classes_dataset/", transform=transforms.Resize((368, 480)))
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=SHUFFLE,
    pin_memory=PIN_MEMORY,
    num_workers=NUM_WORKERS,
    prefetch_factor=PREFETCH_FACTOR,
)


weights = torch.zeros(5).to(DEVICE)
for batch_feats, batch_outs, name in loader:
    for i in range(0, 5):
        weights[i] += torch.sum(batch_outs == i)

weights = 1 / weights
weights = weights / weights.sum().item()

model = UNet()
print(f"Using device {DEVICE}")
model = model.to(DEVICE)
optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

try:
    for i in range(EPOCHS):
        loss_avg = 0
        for batch_feats, batch_outs, name in loader:
            batch_feats, batch_outs = batch_feats.to(DEVICE)/255, batch_outs.to(DEVICE)

            A = model(batch_feats)
            loss = loss_fn(A, TF.resize(batch_outs, size=A.shape[2:]))

            optim.zero_grad()
            loss.backward()
            
            optim.step()
            loss_avg += loss.item()
            losses.append(loss.item())
            
            line.set_data(range(1, len(losses) + 1), losses)
            loss_text.set_text(f"Current Loss: {losses[-1]}")

            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()

        loss_avg /= len(loader)

        print(f"EPOCH {i+1}, COST = {loss_avg}")

except KeyboardInterrupt:
    print("Haulting training.")

finally:
    print(f"MODEL TRAINED!\nSaved at {SAVE_DIR}")
    torch.save(model.state_dict(), SAVE_DIR)

plt.ioff()
plt.show()
