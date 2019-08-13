import os
import argparse
import torch
import fastai.text as FT

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('corpus')
    parser.add_argument('-d', '--output_dir', default='data')
    parser.add_argument('-f', '--output_file', default='data.pkl')
    parser.add_argument('-v', '--vocab_file', default='vocab.pkl')
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    args = parser.parse_args()
    return args

if __name__=='__main__':
    args = get_args()

    tokenizer = FT.Tokenizer(lang='en')
    print("Processing data...")
    data_lm = FT.TextLMDataBunch.from_csv(
        './', args.corpus,                               
        tokenizer=tokenizer, 
        bs=args.batch_size,
        header=None,
        text_cols=0, 
        label_cols=None,
        min_freq=100,
        max_vocab=30000,
        include_bos=True,
        include_eos=True,
        chunksize=10000
    )

    output_path = os.path.join(args.output_dir, args.output_file)
    data_lm.save(output_path)
    print("Save data to %s" % output_path)

    vocab_path =  os.path.join(args.output_dir, args.vocab_file)
    torch.save(data_lm.vocab, vocab_path)
    print("Save vocab to %s" % vocab_path)

    print("Done!")
    