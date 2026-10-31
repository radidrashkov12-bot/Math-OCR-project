import editdistance
import torch
from datasets import load_dataset
from torchvision import transforms
from tqdm import tqdm
from src.model import MathOCR
from src.tokenizer import Tokenizer

def evaluate_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Evaluating on device: {device}')

    # 1. Load Tokenizer & Rebuild Vocab (must match the training slice to get vocab size 93)
    print('Loading dataset and building tokenizer...')
    vocab_ds = load_dataset('deepcopy/MathWriting-human', split='train[110000:130000]')
    all_latex = [item['latex'] for item in vocab_ds]

    tokenizer = Tokenizer()
    tokenizer.build_vocab(all_latex)
    vocab_size = tokenizer.vocab_size()
    print(f'Tokenizer rebuilt with vocabulary size: {vocab_size}')

    # 2. Load Model & Weights
    model = MathOCR(vocab_size=vocab_size, hidden_dim=256).to(device)
    model.load_state_dict(torch.load('math_ocr_model_135k.pth', map_location=device))
    model.eval()

    # 3. Load Held-Out Test Slice (e.g., 500 samples *after* your training data)
    test_ds = load_dataset('deepcopy/MathWriting-human', split='train[135000:135500]')
    print(f'Loaded {len(test_ds)} test samples for evaluation.')

    # Preprocessing transforms (must match training/inference)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((64, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    sos_idx = tokenizer.char_to_idx[tokenizer.start_token]
    eos_idx = tokenizer.char_to_idx[tokenizer.end_token]

    total_cer = 0.0
    exact_matches = 0
    num_samples = len(test_ds)

    print('Starting evaluation loop...')
    with torch.no_grad():
        for i in tqdm(range(num_samples)):
            sample = test_ds[i]
            image = sample['image'].convert('RGB')
            ground_truth = sample['latex']

            # Preprocess image
            image_tensor = transform(image).unsqueeze(0).to(device)

            # Autoregressive Greedy Decoding
            generated_tokens = [sos_idx]
            memory = model.encoder(image_tensor)

            for _ in range(128):
                tgt_tensor = torch.tensor([generated_tokens], dtype=torch.long).to(device)
                logits = model.decoder(tgt_tensor, memory)
                next_token = torch.argmax(logits[0, -1, :]).item()

                generated_tokens.append(next_token)
                if next_token == eos_idx:
                    break

            # Decode prediction
            prediction = tokenizer.decode(generated_tokens)

            # Clean up strings (strip whitespace for fair comparison)
            pred_clean = prediction.strip()
            gt_clean = ground_truth.strip()

            # Compute Exact Match
            if pred_clean == gt_clean:
                exact_matches += 1

            # Compute Character Error Rate (CER)
            edit_dist = editdistance.eval(pred_clean, gt_clean)
            gt_len = max(len(gt_clean), 1)  # Prevent division by zero
            cer = edit_dist / gt_len
            total_cer += cer

    # 4. Calculate Final Metrics
    avg_cer = total_cer / num_samples
    accuracy = (exact_matches / num_samples) * 100

    print('\n' + '=' * 40)
    print('          EVALUATION RESULTS          ')
    print('=' * 40)
    print(f'Total Test Samples Evaluated: {num_samples}')
    print(f'Exact Match Accuracy        : {accuracy:.2f}%')
    print(f'Average Character Error Rate: {avg_cer:.4f}')
    print('=' * 40)

if __name__ == "__main__":
    evaluate_model()