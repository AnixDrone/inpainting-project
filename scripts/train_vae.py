from img_data_class import ImageDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from torch import nn
from model_class import InpaintingVAE
import torch.optim as optim
import time
import tqdm
import argparse
import os


def create_if_not_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('-epochs', '--epochs', type=int,
                        required=True, help="Epochs to train")
    parser.add_argument('-batch', '--batch_size', type=int,
                        required=True, help="Size of a single batch")
    parser.add_argument('-latent', '--latent_size', type=int,
                        required=True, help="Size of the latent space")
    parser.add_argument('-lr', '--learning_rate', type=float,
                        default=1e-3, help="Learning rate")
    parser.add_argument('-train', '--train_path', type=str,
                        required=True, help="Path to the training data")
    parser.add_argument('-test', '--test_path', type=str,
                        required=True, help="Path to the test data")

    return parser.parse_args()


def loss_function(recon_x, x, mu, log_var):
    bce = nn.CrossEntropyLoss(reduction='sum')
    BCE = bce(recon_x.flatten(1), x.flatten(1))
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD


def augment_image(random_erase_fn, img):
    img_list = [random_erase_fn(i) for i in img]
    return torch.stack(img_list)


if __name__ == '__main__':
    args = parse_arg()
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LATENT_SIZE = args.latent_size
    LEARNING_RATE = args.learning_rate
    TRAIN_PATH = args.train_path
    TEST_PATH = args.test_path
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S")

    tr = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ],
    )
    random_erase_tr = transforms.RandomErasing(1)

    train = ImageDataset(TRAIN_PATH, tr)
    train_dl = DataLoader(train, BATCH_SIZE, shuffle=True)

    test = ImageDataset(TEST_PATH, tr)
    test_dl = DataLoader(test, BATCH_SIZE, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = InpaintingVAE(3, LATENT_SIZE, device).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_epoch_list = []
    test_epoch_list = []
    for epoch in range(EPOCHS):
        train_epoch_loss = 0
        test_epoch_loss = 0
        model.train()
        for batch_idx, img in tqdm.tqdm(enumerate(train_dl), total=len(train_dl)):
            input_img = augment_image(random_erase_tr, img).to(device)
            img = img.to(device)

            optimizer.zero_grad()
            recon_img, mu, log_var = model(input_img)
            loss = loss_function(recon_img, img, mu, log_var)
            train_epoch_loss += loss.item()
            loss.backward()
            optimizer.zero_grad()

        with torch.no_grad():
            model.eval()
            for batch_idx, img in enumerate(test_dl):
                img = img.to(device)
                input_img = augment_image(random_erase_tr, img).to(device)

                recon_img, mu, log_var = model(input_img)
                loss = loss_function(recon_img, img, mu, log_var)
                test_epoch_loss += loss.item()

        train_epoch_list.append(train_epoch_loss/len(train_dl))
        test_epoch_list.append(test_epoch_loss/len(test_dl))
        create_if_not_exists("models")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_epoch_list,
            'test_loss': test_epoch_list,
            'latent_size': LATENT_SIZE,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE,
            'parameter_epochs': EPOCHS,
        }, f"models/vae_{timestamp}.pt")
