import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

def preprocess_with_padding(img_path, target_size=(256, 64)):
    """
    Преоразмерява изображението, като запазва неговата пропорция (aspect ratio)
    и добавя бял фон (padding), за да достигне целевия размер (Ширина, Височина).
    """
    img = Image.open(img_path).convert('RGB')
    bg_color = (255, 255, 255)

    img.thumbnail((target_size[0], target_size[1]), Image.Resampling.LANCZOS)

    # Създаваме ново празно бяло изображение с точния целеви размер
    new_img = Image.new("RGB", target_size, bg_color)
    # Поставяме преоразмереното изображение в центъра
    paste_position = (
        (target_size[0] - img.size[0]) // 2,
        (target_size[1] - img.size[1]) // 2
    )
    new_img.paste(img, paste_position)
    return new_img

def predict_with_repetition_penalty(image_path, model, tokenizer, repetition_penalty=1.3, max_len=128):
    """
    Генерира LaTeX код от изображение с правилна обработка на пропорциите и
    наказание за повтарящи се символи.
    """
    device = next(model.parameters()).device
    padded_img = preprocess_with_padding(image_path)

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    image_tensor = transform(padded_img).unsqueeze(0).to(device)

    sos_idx = tokenizer.char_to_idx[tokenizer.start_token]
    eos_idx = tokenizer.char_to_idx[tokenizer.end_token]
    generated_tokens = [sos_idx]

    with torch.no_grad():
        memory = model.encoder(image_tensor)

        for step in range(max_len):
            tgt_tensor = torch.tensor([generated_tokens], dtype=torch.long).to(device)
            logits = model.decoder(tgt_tensor, memory)
            next_token_logits = logits[0, -1, :].clone()

            # Налагане на наказание за вече генерирани токени
            for token in set(generated_tokens):
                if token in [sos_idx, tokenizer.char_to_idx.get('\\', -1)]:
                    continue
                if next_token_logits[token] > 0:
                    next_token_logits[token] /= repetition_penalty
                else:
                    next_token_logits[token] *= repetition_penalty

            next_token = torch.argmax(next_token_logits).item()
            generated_tokens.append(next_token)

            if next_token == eos_idx:
                break

    return tokenizer.decode(generated_tokens)