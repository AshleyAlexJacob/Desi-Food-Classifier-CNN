import torch
from torch.utils.data import DataLoader
import os
class Evaluator:
    # Evaluation loop
    def __init__(self, batch_size:int, data: DataLoader, model, device):
        self.batch_size = batch_size
        self.data = data
        self.model = model
        self.device = device
        
    def start_evaluation_loop(self, epoch):
        try:
            self.model.eval()
            correct = 0
            total = 0
            validation_losses = []
            for batch, (x, y) in enumerate(self.data):
                with torch.no_grad():
                    prediction = self.model(x.to(torch.device(self.device)))
                    validation_loss = self.loss(prediction, y.to(torch.device(self.device)))
                    validation_losses.append(validation_loss.item())
                    # predicted class
                    _, predicted = torch.max(prediction.data, 1)
                    # ground class
                    total += y.size(0)
                    correct += (predicted == y).sum().item()

                    if batch % 5 == 0:
                        print(f"Validation-> Epoch {epoch} -> Batch No{batch}: {validation_loss.item():.4f}")
            epoch_acc = 100. * correct / total
            average_epoch_validation_loss = sum(validation_losses/len(validation_losses))
            print(f"Average Epoch Validation Loss {epoch} -> {average_epoch_validation_loss}")
            print(f"Validation-> Epoch {epoch}: {epoch_acc}")
            return average_epoch_validation_loss, validation_losses, epoch_acc
        
        except Exception as e:
            print(f"Error in Validation Loop Epoch {epoch} and Batch No {batch} due to {e}")
            return None
        
        