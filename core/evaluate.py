import torch
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from dataset.vocab import encode_sentence

def translate_sentence(model, sentence, en_vocab, hi_vocab, src_pad_idx, tgt_pad_idx, device, max_len=50):
    model.eval()
    tokens = encode_sentence(sentence, en_vocab, max_len=max_len)
    src_tensor = torch.tensor(tokens).unsqueeze(0).to(device)

    tgt_tokens = [hi_vocab["<sos>"]]
    for _ in range(max_len):
        tgt_tensor = torch.tensor(tgt_tokens).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(src_tensor, tgt_tensor, src_pad_idx, tgt_pad_idx)
        next_token = output[0, -1].argmax().item()
        tgt_tokens.append(next_token)
        if next_token == hi_vocab["<eos>"]:
            break

    translated = [hi_vocab.itos[idx] for idx in tgt_tokens[1:-1]]
    return ' '.join(translated)

def evaluate_bleu_nltk(model, dataset_subset, en_vocab, hi_vocab, src_pad_idx, tgt_pad_idx, device, max_len=50):
    smoothie = SmoothingFunction().method4
    references = []
    hypotheses = []

    for en_sentence, hi_sentence in dataset_subset:
        pred = translate_sentence(model, en_sentence, en_vocab, hi_vocab, src_pad_idx, tgt_pad_idx, device, max_len)
        pred_tokens = pred.split()
        ref_tokens = hi_sentence.split()

        references.append([ref_tokens])
        hypotheses.append(pred_tokens)

    try:
        score = corpus_bleu(references, hypotheses, smoothing_function=smoothie)
    except Exception:
        score = 0.0
    return score
