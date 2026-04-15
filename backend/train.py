# =====================================
# 1. IMPORTS
# =====================================
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from collections import Counter
from model import LungCNN

# =====================================
# 2. PATHS
# =====================================
train_dir = "data/train"
valid_dir = "data/valid"
test_dir  = "data/test"

# =====================================
# 3. TRANSFORMS
# =====================================
train_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128,128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

valid_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# =====================================
# 4. LOAD DATA
# =====================================
train_data = datasets.ImageFolder(train_dir, transform=train_transform)
valid_data = datasets.ImageFolder(valid_dir, transform=valid_transform)
test_data  = datasets.ImageFolder(test_dir, transform=valid_transform)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True, drop_last=True)
valid_loader = DataLoader(valid_data, batch_size=16)
test_loader  = DataLoader(test_data, batch_size=16)

print("Classes:", train_data.classes)

# =====================================
# 5. CLASS DISTRIBUTION
# =====================================
print("\nClass Distribution:")
print("Train:", Counter(train_data.targets))
print("Valid:", Counter(valid_data.targets))
print("Test :", Counter(test_data.targets))

# =====================================
# 6. CLASS WEIGHTS (FIX IMBALANCE)
# =====================================
class_counts = Counter(train_data.targets)
total_samples = sum(class_counts.values())

class_weights = []
for i in range(len(class_counts)):
    class_weights.append(total_samples / class_counts[i])

class_weights = torch.tensor(class_weights, dtype=torch.float32)

print("Class Weights:", class_weights)

# =====================================
# 7. MODEL
# =====================================
model = LungCNN()

# =====================================
# 8. LOSS + OPTIMIZER
# =====================================
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# =====================================
# 9. TRAINING WITH EARLY STOPPING
# =====================================
best_acc = 0
patience = 5
counter = 0

for epoch in range(20):

    # ---- TRAIN ----
    model.train()
    train_loss = 0

    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    # ---- VALIDATION ----
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in valid_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = 100 * correct / total

    print(f"\nEpoch {epoch+1}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.2f}%")

    # ---- SAVE BEST MODEL ----
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "lung_model.pth")
        print("✅ Best model saved")
        counter = 0
    else:
        counter += 1

    # ---- EARLY STOPPING ----
    if counter >= patience:
        print("⛔ Early stopping triggered")
        break

# =====================================
# 10. FINAL TEST EVALUATION
# =====================================
model.load_state_dict(torch.load("lung_model.pth"))
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_acc = 100 * correct / total
print(f"\n🔥 Final Test Accuracy: {test_acc:.2f}%")