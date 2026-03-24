def collate_batch(batch, tokenizer):
    padded = tokenizer.pad(
        batch,
        padding=True,
        return_tensors="pt",
    )
    padded["labels"] = padded["input_ids"].clone()
    return padded