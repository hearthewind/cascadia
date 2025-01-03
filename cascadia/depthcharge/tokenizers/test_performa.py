import torch
from pyteomics import proforma

from cascadia.depthcharge.tokenizers import PeptideTokenizer

if __name__ == '__main__':
    raw_seq = 'M[Oxidation]IL'
    # pep, meta = proforma.parse(raw_seq)

    tokenizer = PeptideTokenizer.from_massivekb(reverse=False, replace_isoleucine_with_leucine=True)
    tokenized = tokenizer.tokenize(raw_seq)

    print('tokenized', tokenized)
    detokenized = tokenizer.detokenize(torch.tensor(tokenized).reshape([1, -1]))
    print('detokenized', detokenized)