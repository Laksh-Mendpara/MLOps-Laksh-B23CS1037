import os
import time
import tempfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import ray
from ray import train, tune
from dataset.data_loader import TranslationDataset, collate_fn
from models.transformer import Transformer
from core.evaluate import evaluate_bleu_nltk

def train_tune(config, df=None, en_vocab=None, hi_vocab=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MAX_LEN = 50
    SRC_PAD_IDX = en_vocab["<pad>"]
    TGT_PAD_IDX = hi_vocab["<pad>"]
    NUM_EPOCHS = config.get("num_epochs", 20)
    
    dataset = TranslationDataset(df, en_vocab, hi_vocab, max_len=MAX_LEN)
    train_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn)

    model = Transformer(
        src_vocab_size=len(en_vocab),
        tgt_vocab_size=len(hi_vocab),
        d_model=config["d_model"],
        num_layers=6, # fixed for baseline
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        max_len=MAX_LEN,
        dropout=config["dropout"]
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=TGT_PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    val_dataset = [
        ("I love you.", "मैं तुमसे प्यार करता हूँ।"),
        ("How are you?", "आप कैसे हैं?"),
        ("You should sleep.", "आपको सोना चाहिए।"),
        ("Maybe Tom doesn't love you.", "टॉम शायद तुमसे प्यार नहीं करता है।"),
        ("Let me tell Tom.","मुझे टॉम को बताने दीजिए।")
    ]

    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        model.train()
        epoch_loss = 0

        for src, tgt_input, tgt_output in train_loader:
            src, tgt_input, tgt_output = src.to(device), tgt_input.to(device), tgt_output.to(device)
            output = model(src, tgt_input, SRC_PAD_IDX, TGT_PAD_IDX)
            output = output.view(-1, output.shape[-1])
            tgt_output = tgt_output.view(-1)

            loss = criterion(output, tgt_output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        bleu_score = evaluate_bleu_nltk(
            model, val_dataset, en_vocab, hi_vocab, SRC_PAD_IDX, TGT_PAD_IDX, device, max_len=MAX_LEN
        )
            
        with tempfile.TemporaryDirectory() as temp_ckpt_dir:
            ckpt_path = os.path.join(temp_ckpt_dir, "model.pth")
            torch.save(model.state_dict(), ckpt_path)
            ray_checkpoint = train.Checkpoint.from_directory(temp_ckpt_dir)
            tune.report(
                {"loss": avg_loss, "bleu": bleu_score, "time": time.time() - start_time}, 
                checkpoint=ray_checkpoint
            )

def train_final_model(config, df, en_vocab, hi_vocab, save_path="b23cs1037_ass_4_best_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training final model on {device} with config:")
    print(config)

    MAX_LEN = 50
    SRC_PAD_IDX = en_vocab["<pad>"]
    TGT_PAD_IDX = hi_vocab["<pad>"]
    
    PATIENCE = 5
    MAX_EPOCHS = 100
    
    dataset = TranslationDataset(df, en_vocab, hi_vocab, max_len=MAX_LEN)
    train_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn)

    model = Transformer(
        src_vocab_size=len(en_vocab),
        tgt_vocab_size=len(hi_vocab),
        d_model=config["d_model"],
        num_layers=6, # fixed for baseline
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        max_len=MAX_LEN,
        dropout=config["dropout"]
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=TGT_PAD_IDX)
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])

    val_dataset = [
        ("I love you.", "मैं तुमसे प्यार करता हूँ।"),
        ("How are you?", "आप कैसे हैं?"),
        ("You should sleep.", "आपको सोना चाहिए।"),
        ("Maybe Tom doesn't love you.", "टॉम शायद तुमसे प्यार नहीं करता है।"),
        ("Let me tell Tom.","मुझे टॉम को बताने दीजिए।")
    ]

    best_bleu = 0.0
    epochs_without_improvement = 0
    start_time = time.time()
    best_loss_at_best_bleu = float('inf')
    
    for epoch in range(MAX_EPOCHS):
        model.train()
        epoch_loss = 0

        for src, tgt_input, tgt_output in train_loader:
            src, tgt_input, tgt_output = src.to(device), tgt_input.to(device), tgt_output.to(device)

            output = model(src, tgt_input, SRC_PAD_IDX, TGT_PAD_IDX)
            output = output.view(-1, output.shape[-1])
            tgt_output = tgt_output.view(-1)

            loss = criterion(output, tgt_output)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        bleu_score = evaluate_bleu_nltk(
            model, val_dataset, en_vocab, hi_vocab, SRC_PAD_IDX, TGT_PAD_IDX, device, max_len=MAX_LEN
        )
        
        print(f"Epoch {epoch+1}/{MAX_EPOCHS} | Loss: {avg_loss:.4f} | BLEU: {bleu_score*100:.2f}")

        if bleu_score > best_bleu:
            best_bleu = bleu_score
            best_loss_at_best_bleu = avg_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), save_path)
            print(f"  --> Saved new best model (BLEU: {best_bleu*100:.2f})")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
                
    total_time = time.time() - start_time
    print(f"Final training completed in {total_time:.2f}s")
    
    return best_bleu, best_loss_at_best_bleu, total_time
