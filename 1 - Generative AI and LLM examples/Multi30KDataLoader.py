# Multi30K Dataset DataLoader for German-English Translation

# Import required libraries
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from typing import Iterable, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Load the Multi30K dataset from Hugging Face
print("Loading Multi30K dataset...")
dataset = load_dataset("bentrevett/multi30k")
print("Dataset loaded successfully!")
print(f"Training samples: {len(dataset['train'])}")
print(f"Validation samples: {len(dataset['validation'])}")
print(f"Test samples: {len(dataset['test'])}")


# Define languages
SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'

# Setup tokenizers for both languages
token_transform = {}
token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='de_core_news_sm')
token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')

# Define special symbols and indices
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

# Helper function to yield tokens from the dataset
def yield_tokens(data_iter: List[dict], language: str) -> Iterable[List[str]]:
    """Yield tokens for building vocabulary"""
    for data_sample in data_iter:
        yield token_transform[language](data_sample[language])

# Build vocabularies for both languages
vocab_transform = {}

for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    print(f"Building vocabulary for {ln}...")
    
    # Convert dataset to list and sort by source sentence length
    train_data = list(dataset['train'])
    sorted_data = sorted(train_data, key=lambda x: len(x[SRC_LANGUAGE].split()))
    
    # Build vocabulary
    vocab_transform[ln] = build_vocab_from_iterator(
        yield_tokens(sorted_data, ln),
        min_freq=1,
        specials=special_symbols,
        special_first=True
    )
    
    # Set default index for unknown tokens
    vocab_transform[ln].set_default_index(UNK_IDX)
    print(f"  Vocabulary size for {ln}: {len(vocab_transform[ln])}")

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Tensor transform for source (with flip option)
def tensor_transform_s(token_ids: List[int]):
    """Add BOS/EOS tokens and flip source sequence"""
    return torch.cat((
        torch.tensor([BOS_IDX]),
        torch.flip(torch.tensor(token_ids), dims=(0,)),
        torch.tensor([EOS_IDX])
    ))

# Tensor transform for target (no flip)
def tensor_transform_t(token_ids: List[int]):
    """Add BOS/EOS tokens to target sequence"""
    return torch.cat((
        torch.tensor([BOS_IDX]),
        torch.tensor(token_ids),
        torch.tensor([EOS_IDX])
    ))

# Helper function to chain transforms
def sequential_transforms(*transforms):
    """Chain multiple transform functions"""
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# PyTorch Dataset wrapper for Multi30K
class Multi30KDataset(Dataset):
    """PyTorch Dataset wrapper for Multi30K from Hugging Face"""
    
    def __init__(self, hf_dataset, src_lang='de', tgt_lang='en'):
        """
        Initialize the dataset wrapper
        
        Args:
            hf_dataset: Hugging Face dataset split
            src_lang: Source language code
            tgt_lang: Target language code
        """
        self.data = list(hf_dataset)
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        
        # Sort by source sentence length for efficiency
        self.data = sorted(self.data, key=lambda x: len(x[src_lang].split()))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        return example[self.src_lang], example[self.tgt_lang]

# Create dataset instances
train_dataset = Multi30KDataset(dataset['train'], SRC_LANGUAGE, TGT_LANGUAGE)
valid_dataset = Multi30KDataset(dataset['validation'], SRC_LANGUAGE, TGT_LANGUAGE)
test_dataset = Multi30KDataset(dataset['test'], SRC_LANGUAGE, TGT_LANGUAGE)

# Initialize text transforms (will be updated in get_translation_dataloaders)
text_transform = {}

# Collate function to process batches
def collate_fn(batch):
    """Collate function to process batches"""
    src_batch, tgt_batch = [], []
    
    for src_sample, tgt_sample in batch:
        # Process source sequences
        src_sequences = text_transform[SRC_LANGUAGE](src_sample.rstrip("\n"))
        src_sequences = torch.tensor(src_sequences, dtype=torch.int64)
        
        # Process target sequences
        tgt_sequences = text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n"))
        tgt_sequences = torch.tensor(tgt_sequences, dtype=torch.int64)
        
        src_batch.append(src_sequences)
        tgt_batch.append(tgt_sequences)
    
    # Pad sequences
    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX, batch_first=True)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX, batch_first=True)
    
    # Transpose to get (seq_len, batch_size)
    src_batch = src_batch.t()
    tgt_batch = tgt_batch.t()
    
    return src_batch.to(device), tgt_batch.to(device)

# Main function to create dataloaders
def get_translation_dataloaders(batch_size=4, flip=False):
    """
    Create DataLoaders for translation task
    
    Args:
        batch_size: Batch size for DataLoaders
        flip: Whether to flip the source sequences
    
    Returns:
        train_dataloader, valid_dataloader
    """
    global text_transform
    
    # Update text_transform based on flip parameter
    if flip:
        text_transform[SRC_LANGUAGE] = sequential_transforms(
            token_transform[SRC_LANGUAGE],
            vocab_transform[SRC_LANGUAGE],
            tensor_transform_s
        )
    else:
        text_transform[SRC_LANGUAGE] = sequential_transforms(
            token_transform[SRC_LANGUAGE],
            vocab_transform[SRC_LANGUAGE],
            tensor_transform_t
        )
    
    text_transform[TGT_LANGUAGE] = sequential_transforms(
        token_transform[TGT_LANGUAGE],
        vocab_transform[TGT_LANGUAGE],
        tensor_transform_t
    )
    
    # Create DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        drop_last=True,
        shuffle=False  # Already sorted by length
    )
    
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        drop_last=True,
        shuffle=False
    )
    
    return train_dataloader, valid_dataloader

# Helper functions for decoding
def index_to_eng(seq_en):
    """Convert English token indices back to text"""
    return " ".join([vocab_transform['en'].get_itos()[index.item()] for index in seq_en])

def index_to_german(seq_de):
    """Convert German token indices back to text"""
    return " ".join([vocab_transform['de'].get_itos()[index.item()] for index in seq_de])

# Test the dataloader
if __name__ == "__main__":
    print("\nCreating DataLoaders...")
    train_loader, valid_loader = get_translation_dataloaders(batch_size=8, flip=False)
    
    print(f"\nTrain DataLoader: {len(train_loader)} batches")
    print(f"Valid DataLoader: {len(valid_loader)} batches")
    
    # Get one batch for testing
    src_batch, tgt_batch = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  Source: {src_batch.shape} (seq_len, batch_size)")
    print(f"  Target: {tgt_batch.shape} (seq_len, batch_size)")
    
    # Decode and display first sample from batch
    src_seq = src_batch[:, 0]
    tgt_seq = tgt_batch[:, 0]
    
    # Remove padding
    src_seq = src_seq[src_seq != PAD_IDX]
    tgt_seq = tgt_seq[tgt_seq != PAD_IDX]
    
    print(f"\nSample translation pair:")
    print(f"  German: {index_to_german(src_seq)}")
    print(f"  English: {index_to_eng(tgt_seq)}")
