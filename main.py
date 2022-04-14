from torch.utils.data import DataLoader

from nn.ConvLSTM import ConvLSTM_MovingMNIST
from utils.data import MovingMNISTDataset
from utils.trainer import Trainer


def train_ConvLSTM_Moving_MNIST():
    convlstm = ConvLSTM_MovingMNIST(in_channels=16, hidden_channels_list=[128, 64, 64], size=(16, 16),
                                    kernel_size_list=[5, 5, 5])
    train_set = MovingMNISTDataset("train")
    test_set = MovingMNISTDataset("test")
    validation_set = MovingMNISTDataset("validation")
    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=8)
    validation_loader = DataLoader(validation_set, batch_size=8)
    trainer = Trainer(max_epoch=1000, device="cuda:0", to_save="results/MovingMNIST/ConvLSTM")
    # trainer.fit(convlstm, train_loader, validation_loader)

    trainer.predict(convlstm, test_loader=test_loader,
                    ckpt_path="results/MovingMNIST/ConvLSTM/checkpoint_000020_0.0200864142_temp.pth")


if __name__ == '__main__':
    train_ConvLSTM_Moving_MNIST()
