import torch
import torch.nn as nn
import torch.optim as optim
from src.dataset import get_dataloaders
from src.model import MathOCR
from src.tokenizer import Tokenizer 

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Load data
    from datasets import load_dataset
    raw_ds = load_dataset('deepcopy/MathWriting-human', split='train[:5000]')
    all_latex = [item['latex'] for item in raw_ds]
    tokenizer = Tokenizer()
    tokenizer.build_vocab(all_latex)
    vocab_size = tokenizer.vocab_size()
    print(f"Tokenizer built with vocabulary size: {vocab_size}")

    train_loader, test_loader = get_dataloaders(tokenizer, batch_size=16)
    model = MathOCR(vocab_size=vocab_size, hidden_dim=256).to(device)
    pad_idx = tokenizer.char_to_idx[tokenizer.pad_token]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=0.001) #learning rate is set to 0.001 by default

    num_epochs = 5
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)
            decoder_input = targets[:, :-1]  # Exclude the last token for decoder input
            target_output = targets[:, 1:]    # Exclude the first token for target output
            optimizer.zero_grad()

            logits = model(images, decoder_input)
            loss = criterion(
          logits.reshape(-1, vocab_size), target_output.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % 50 == 0:
             print(
            f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}'
            )
        avg_loss = total_loss / len(train_loader)
        print(f'---> Epoch {epoch+1} Completed. Average Loss: {avg_loss:.4f}')
    torch.save(model.state_dict(), 'math_ocr_model.pth')
    print("Model training completed and saved as 'math_ocr_model.pth'.")

if __name__ == "__main__": 
   train_model()
      
