import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
from d2l import torch as d2l
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim.lr_scheduler import StepLR

print("PyTorch Version: ", torch.__version__)

# Device configuration - check for CUDA availability
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Instantiate model and move to device
model = hsdnet()
model.to(device)

# Set loss function
loss_fn = nn.CrossEntropyLoss()

# Set optimizer with weight decay
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)

# Learning rate scheduler
scheduler = StepLR(optimizer, step_size=10, gamma=0.2)

# Evaluation function with MC Dropout for uncertainty estimation
def evaluate_accuracy_gpu(net, data_iter, loss_fn, device=None):
    """Compute model accuracy on dataset using GPU with MC Dropout"""
    net.eval()  # Set to evaluation mode
    # Enable dropout layers for MC Dropout
    for module in net.modules():
        if module.__class__.__name__.startswith("Dropout"):
            module.train()
    
    if not device:
        device = next(iter(net.parameters())).device
        
    mc_samples = 100  # Number of Monte Carlo samples
    metric = d2l.Accumulator(3)  # For loss, accuracy, count
    
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            
            # MC Dropout: collect multiple predictions
            predictions = []
            for _ in range(mc_samples):
                output = net(X)
                predictions.append(output)
                
            # Average predictions
            predictions = torch.stack(predictions, dim=0)
            avg_prediction = torch.mean(predictions, dim=0)
            
            # Calculate loss
            l = loss_fn(avg_prediction, y.long())
            
            # Calculate accuracy
            y_hat = avg_prediction.argmax(axis=1)
            correct = (y_hat == y).sum().item()
            
            metric.add(l * X.shape[0], correct, X.shape[0])
            
    return metric[0] / metric[2], metric[1] / metric[2]  # Avg loss, accuracy


# Enhanced evaluation with full metrics and visualizations
def evaluate_accuracy_gpu_test(net, data_iter, loss_fn, device=None):
    """Comprehensive evaluation with metrics and visualizations"""
    net.eval()  # Set to evaluation mode
    # Enable dropout layers for MC Dropout
    for module in net.modules():
        if module.__class__.__name__.startswith("Dropout"):
            module.train()
    
    if not device:
        device = next(iter(net.parameters())).device
    
    mc_samples = 100  # Number of Monte Carlo samples
    metric = d2l.Accumulator(3)  # For loss, accuracy, count
    confusion_mat = torch.zeros(2, 2, dtype=torch.int64)  # Confusion matrix
    
    # Containers for metrics
    y_trues = []
    y_preds = []
    class_probs = {0: [], 1: []}  # Class probability containers
    
    with torch.no_grad():
        for X, y in data_iter:
            X, y = X.to(device), y.to(device)
            
            # MC Dropout: collect multiple predictions
            predictions = []
            for _ in range(mc_samples):
                output = net(X)
                predictions.append(output)
                
            # Average predictions
            predictions = torch.stack(predictions, dim=0)
            avg_prediction = torch.mean(predictions, dim=0)
            
            # Calculate loss
            l = loss_fn(avg_prediction, y.long())
            
            # Process predictions
            softmax_preds = F.softmax(avg_prediction, dim=1)
            y_hat = softmax_preds.argmax(axis=1)
            correct = (y_hat == y).sum().item()
            
            # Update metrics
            metric.add(l * X.shape[0], correct, X.shape[0])
            
            # Collect for visualization
            y_trues.extend(y.cpu().numpy())
            y_preds.extend(softmax_preds.cpu().numpy())
            class_probs[0].extend(softmax_preds[:, 0].cpu().numpy())
            class_probs[1].extend(softmax_preds[:, 1].cpu().numpy())
            
            # Update confusion matrix
            for true, pred in zip(y.cpu(), y_hat.cpu()):
                confusion_mat[true, pred] += 1
                
    # Calculate metrics
    test_loss = metric[0] / metric[2]
    accuracy = metric[1] / metric[2]
    precision, recall, f1_score = calculate_metrics(confusion_mat, 2)

    # Create visualizations
    class_names = ["non-Hsd", "Hsd"]
    fig, axs = plt.subplots(1, 3, figsize=(19, 5))
    plt.rcParams.update({'font.size': 14})
    
    # ROC Curve
    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_trues, class_probs[i], pos_label=i)
        roc_auc = auc(fpr, tpr)
        axs[0].plot(fpr, tpr, lw=2, 
                   label=f'ROC for {name} (AUC = {roc_auc:.2f})')
    axs[0].plot([0, 1], [0, 1], 'k--', lw=2)
    axs[0].set_xlabel('False Positive Rate')
    axs[0].set_ylabel('True Positive Rate')
    axs[0].legend(loc="lower right")
    
    # Precision-Recall Curve
    for i, name in enumerate(class_names):  
        precision_vals, recall_vals, _ = precision_recall_curve(
            (np.array(y_trues) == i), class_probs[i])
        pr_auc = auc(recall_vals, precision_vals)
        axs[1].plot(recall_vals, precision_vals, 
                   label=f'{name} (AUC = {pr_auc:.2f})')
    axs[1].set_xlabel('Recall')
    axs[1].set_ylabel('Precision')
    axs[1].legend(loc="lower left")
    
    # Confusion Matrix
    sns.heatmap(confusion_mat.numpy(), annot=True, fmt='d', 
               cmap='Purples', ax=axs[2])
    axs[2].set_xlabel('Predicted labels')
    axs[2].set_ylabel('True labels')
    axs[2].set_xticklabels(class_names)
    axs[2].set_yticklabels(class_names, rotation=0)
    
    plt.tight_layout()
    
    return test_loss, accuracy, precision, recall, f1_score


# Calculate precision, recall, and F1 score
def calculate_metrics(confusion_mat, num_classes):
    """Compute precision, recall, and F1 score from confusion matrix"""
    metrics = {
        'precision': torch.zeros(num_classes),
        'recall': torch.zeros(num_classes),
        'f1': torch.zeros(num_classes)
    }
    
    for i in range(num_classes):
        tp = confusion_mat[i, i]
        fp = confusion_mat[:, i].sum() - tp
        fn = confusion_mat[i, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics['precision'][i] = precision
        metrics['recall'][i] = recall
        metrics['f1'][i] = f1
        
    return metrics['precision'], metrics['recall'], metrics['f1']


# Training loop
best_valid_loss = float('inf')
model_save_path = r"D:\ZYH\code\model\binary\best_network.pth"

print(f"Starting training for {num_epochs} epochs...")
for epoch in range(num_epochs):
    epoch_start = time.time()
    metric = d2l.Accumulator(3)  # For loss, accuracy, count
    model.train()
    
    # Training phase with progress bar
    for X, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        optimizer.zero_grad()
        X, y = X.to(device), y.to(device)
        
        # Forward pass
        y_hat = model(X)
        l = loss_fn(y_hat, y.long())
        
        # Backward pass and optimize
        l.backward()
        optimizer.step()
        
        # Update metrics
        with torch.no_grad():
            acc = d2l.accuracy(y_hat, y)
            metric.add(l * X.shape[0], acc, X.shape[0])
    
    # Calculate training metrics
    train_loss = metric[0] / metric[2]
    train_acc = metric[1] / metric[2]
    
    # Validation phase
    valid_loss, valid_acc = evaluate_accuracy_gpu(model, valid_loader, loss_fn)
    
    # Update learning rate
    scheduler.step()
    
    # Timing and logging
    epoch_time = time.time() - epoch_start
    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
          f"Valid Loss: {valid_loss:.4f}, Acc: {valid_acc:.4f} | "
          f"Time: {epoch_time:.1f}s")
    
    # Save best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), model_save_path)
        print(f"Saved new best model with validation loss: {valid_loss:.4f}")