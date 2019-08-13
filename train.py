import os
import argparse
import torch
import fastai.text as FT

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_lm')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_cycle', type=int, default=1)
    parser.add_argument('--max_lr', type=float, default=1e-2)
    parser.add_argument('--load_model')
    
    args = parser.parse_args()
    return args


if __name__=='__main__':
    args = get_args()
    data_file = args.data_lm
    batch_size = args.batch_size
    num_cycle = args.num_cycle
    max_lr = args.max_lr
    weight_file = args.load_model

    data_lm = FT.load('./', data_file)
    learner = FT.language_model_learner(data_lm, FT.AWD_LSTM)
    
    if not os.path.isdir(weight_file):
        raise Exception("Invalid weight path")
    else:
        print("Load weight")
        learner = learner.load(weight_file)
    
    learner.fit_one_cycle(num_cycle, max_lr)
    learner.save("model")
    learner.save_encoder("encoder")


