import torch
from src.data.loader import train_loader, test_loader
from src.model.cnn import CNN
from src.model.train import Trainer
from src.model.evaluation import Evaluator

def main():
    try:

        # Training Config
        EPOCHS = 100
        BATCH_SIZE = 32
        LEARNING_RATE = 0.001
        DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        my_model = CNN()
        print("Using device:", DEVICE)
        torch.set_default_device(DEVICE)

        model_trainer = Trainer(
            batch_size= BATCH_SIZE,
            learning_rate= LEARNING_RATE,
            data= train_loader,
            model = my_model,
            model_path="my_cnn",
            device= DEVICE
        )

        model_evaluator = Evaluator(
            batch_size= BATCH_SIZE,
            data= test_loader,
            model = my_model,
        )

        # epoch loop

        for epoch in EPOCHS:
            # run training loop
            average_epoch_training_loss, training_losses, epoch_training_acc = model_trainer.start_training_loop(epoch)

            # run validation loop
            average_epoch_validation_loss, validation_losses, epoch_validation_acc = model_evaluator.start_evaluation_loop(epoch)


    except Exception as e:
        pass