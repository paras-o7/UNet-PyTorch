import torch
from torch.utils.data import DataLoader, Subset
from semdronedataset import SemanticDroneDataset
from unet import UNet
import matplotlib.pyplot as plt
# from torchvision import transforms
import torchvision.transforms.functional as TF

# HYPERPARAMETERS
EPOCHS = 100
BATCH_SIZE = 2
LEARNING_RATE = 3e-4
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.xpu.is_available():
    DEVICE = "xpu"
else:
    DEVICE = "cpu"
SHUFFLE = True
PIN_MEMORY = True
SAVE_DIR = "./model_save/binary.pth"
SAVE_DIR_BEST = "./model_save/binary_best.pth"
NUM_WORKERS = 4
PREFETCH_FACTOR = 4


plt.ion()
fig, ax = plt.subplots()
losses = []
(line,) = ax.plot([], [], linewidth=2)
ax.set_xlabel("Iteration")
ax.set_ylabel("Cost")
ax.set_title(f"Training Loss | LR = {LEARNING_RATE}")
loss_text = ax.text(0.02, 0.95, "", transform=ax.transAxes, fontsize=10, verticalalignment="top")


dataset = SemanticDroneDataset(
    "./archive/classes_dataset/classes_dataset/",
    # transform=transforms.Resize((368, 480), interpolation=TF.InterpolationMode.NEAREST),
)
trainset = Subset(dataset, indices=list(range(300)))
testset = Subset(dataset, indices=list(range(300, 400)))
train_loader = DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=SHUFFLE,
    pin_memory=PIN_MEMORY,
    num_workers=NUM_WORKERS,
    prefetch_factor=PREFETCH_FACTOR,
)
test_loader = DataLoader(
    testset,
    batch_size=1,
    shuffle=False,
    pin_memory=PIN_MEMORY,
    num_workers=NUM_WORKERS,
    prefetch_factor=PREFETCH_FACTOR,
)



weights = torch.zeros(5).to(DEVICE)
for batch_feats, batch_outs, name in train_loader:
    for i in range(0, 5):
        weights[i] += torch.sum(batch_outs == i)

weights = weights.sum() / weights
weights = weights / weights.mean()
print(weights)

model = UNet()
print(f"Using device {DEVICE}")
model = model.to(DEVICE)
optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

best = (1000000, model.state_dict())
try:
    for i in range(EPOCHS):
        loss_avg = 0
        for t, (batch_feats, batch_outs, name) in enumerate(train_loader):
            batch_feats, batch_outs = batch_feats.to(DEVICE) / 255, batch_outs.to(DEVICE)

            A = model(batch_feats)
            loss = loss_fn(A, TF.center_crop(batch_outs, output_size=A.shape[2:]))

            loss.backward()

            optim.step()
            optim.zero_grad()

            loss_avg += loss.item()
            losses.append(loss.item())

            line.set_data(range(1, len(losses) + 1), losses)
            loss_text.set_text(f"Current Loss: {losses[-1]:.6f}")
            
            if loss.item() < best[0]:
                best = (loss.item(), model.state_dict())

            ax.relim()
            ax.autoscale_view()
            fig.canvas.draw()
            fig.canvas.flush_events()
            
            if (t + 1) % 5 == 0:
                torch.save(model.state_dict(), SAVE_DIR)

        loss_avg /= len(train_loader)

        print(f"EPOCH {i+1}, COST = {loss_avg}")

except KeyboardInterrupt:
    print("Haulting training.")

finally:
    print(f"MODEL TRAINED!\nSaved at {SAVE_DIR}, best loss case saved at: {SAVE_DIR_BEST}")
    torch.save(model.state_dict(), SAVE_DIR)
    torch.save(best[1], SAVE_DIR_BEST)

plt.ioff()
plt.show()
