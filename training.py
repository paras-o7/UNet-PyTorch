import torch
from torch.utils.data import DataLoader
from semdronedataset import SemanticDroneDataset
from unet import UNet
import matplotlib.pyplot as plt


EPOCHS = 100
BATCH_SIZE = 1
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
NUM_WORKERS = 8
PREFETCH_FACTOR = 2


dataset = SemanticDroneDataset("./archive/classes_dataset/classes_dataset/")
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=SHUFFLE,
    pin_memory=PIN_MEMORY,
    num_workers=NUM_WORKERS,
    prefetch_factor=PREFETCH_FACTOR,
)

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

model = UNet()
print(f"Using device {DEVICE}")
model = model.to(DEVICE)
optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.CrossEntropyLoss()

for i in range(EPOCHS):
    loss_avg = 0
    for batch_feats, batch_outs in loader:
        batch_feats, batch_outs = batch_feats.to(DEVICE), batch_outs.to(DEVICE)
        # print(batch_outs)
        # print(batch_outs.min(), batch_outs.max())
        
        batch_feats /= 255

        A = model(batch_feats)
        # A.squeeze_(1)
        # print(A.shape, batch_outs.shape)
        loss = loss_fn(A, batch_outs)

        # print('LOSSFN')

        optim.zero_grad()
        loss.backward()

        # print("backprop")
        optim.step()
        loss_avg += loss.item()
        losses.append(loss.item())
        # print(loss.item())
        # ---- update plot ----
        line.set_data(range(1, len(losses) + 1), losses)
        loss_text.set_text(f"Current Loss: {losses[-1]}")

        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw()
        fig.canvas.flush_events()

    loss_avg /= len(loader)



    print(f"EPOCH {i+1}, COST = {loss_avg}")

plt.ioff()
plt.show()

print(f"MODEL TRAINED!\nSaved at {SAVE_DIR}")
torch.save(model.state_dict(), SAVE_DIR)
