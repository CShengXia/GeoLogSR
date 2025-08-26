import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from config import EPOCHS, LEARNING_RATE, EARLY_STOP_PATIENCE, LOSS_WEIGHTS, DEVICE
from losses import EnhancedCombinedLoss

def train_model(model, train_loader, val_loader):
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
    loss_fn = EnhancedCombinedLoss(
        alpha=LOSS_WEIGHTS['mse'],
        beta=LOSS_WEIGHTS['mae'],
        gamma=LOSS_WEIGHTS['ssim'],
        delta=LOSS_WEIGHTS['gradient']
    )
    best_val_loss = float('inf')
    patience_counter = 0
    for epoch in range(EPOCHS):
        model.train()
        train_loss_sum = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss, mse, mae, ssim_loss, grad_loss = loss_fn(y_pred, y_batch)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_sum += loss.item()
        scheduler.step()
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                y_pred = model(x_batch)
                loss, mse, mae, ssim_loss, grad_loss = loss_fn(y_pred, y_batch)
                val_loss_sum += loss.item()
        avg_train_loss = train_loss_sum / len(train_loader)
        avg_val_loss = val_loss_sum / len(val_loader)
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
    model.load_state_dict(torch.load("best_model.pth"))
    return model