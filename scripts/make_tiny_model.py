"""Create a tiny, randomly initialised DistilBERT and tokenizer from the training texts (no download).

Used by the tests and for checking the whole pipeline offline; it is NOT the model behind the reported results.
    python -m scripts.make_tiny_model /tmp/tiny-distilbert
"""
import sys
from pathlib import Path

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors, trainers
from transformers import DistilBertConfig, DistilBertModel, DistilBertTokenizerFast

from qasd.data import splits


def main(out: str) -> Path:
    path = Path(out)
    path.mkdir(parents=True, exist_ok=True)
    texts = [t for t, _ in splits()["train"]]
    tk = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tk.normalizer = normalizers.BertNormalizer(lowercase=True)
    tk.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tk.train_from_iterator(texts, trainers.WordPieceTrainer(vocab_size=2000, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]))
    tk.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B [SEP]",
                                                      special_tokens=[("[CLS]", tk.token_to_id("[CLS]")), ("[SEP]", tk.token_to_id("[SEP]"))])
    tok = DistilBertTokenizerFast(tokenizer_object=tk, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]",
                                  sep_token="[SEP]", mask_token="[MASK]", model_max_length=128)
    tok.save_pretrained(path)
    cfg = DistilBertConfig(vocab_size=tk.get_vocab_size(), dim=64, n_layers=2, n_heads=2, hidden_dim=128, max_position_embeddings=128)
    DistilBertModel(cfg).save_pretrained(path)
    return path


if __name__ == "__main__":
    print(main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/tiny-distilbert"))
