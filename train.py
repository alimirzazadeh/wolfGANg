import os
import argparse
import numpy as np
from matplotlib import pyplot as plt

import torch
from torch import nn
from torch.utils.data import DataLoader
from musegan import MuseGAN
from data.utils import MidiDataset
from ipdb import set_trace as bp
from data.utils import postProcess
from inspirational_generation import InspirationalGeneration

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog = 'top', description='Train MusaGAN.')
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
    parser.add_argument("--z_dimension", type=int, default=32, help="Z(noise)-space dimension.")
    parser.add_argument("--g_channels", type=int, default=1024, help="Generator hidden channels.")
    parser.add_argument("--g_features", type=int, default=1024, help="Generator hidden features.")
    parser.add_argument("--g_lr", type=float, default=0.001, help="Generator learning rate.")
    parser.add_argument("--c_channels", type=int, default=128, help="Critic hidden channels.")
    parser.add_argument("--c_features", type=int, default=1024, help="Critic hidden features.")
    parser.add_argument("--c_lr", type=float, default=0.001, help="Critic learning rate.")
    args = parser.parse_args()
    # parameters of musegan
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    gan_args = args.__dict__.copy()
    gan_args.pop('epochs', None)
    gan_args.pop('batch_size', None)
    gan_args["device"] = device
    # train
    print("Start training ...")
    print("Loading dataset ...")
    dataset = MidiDataset(path='data/chorales/Jsb16thSeparated.npz')
    #bp()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
    print("Loading model ...")
    musegan = MuseGAN(**gan_args)
    print("Start training ...")
    finalgen, finalcritic = musegan.train(dataloader=dataloader, epochs=args.epochs)
    print("Training finished.")
    finalgen = finalgen.eval()
    batch_size = 1


    # generate MIDI files components in a for loop

    generating = False

    if generating:
        for i in range(3):
            cords = torch.randn(batch_size, 32).to(device)
            style = torch.randn(batch_size, 32).to(device)
            melody = torch.randn(batch_size, 4, 32).to(device)
            groove = torch.randn(batch_size, 4, 32).to(device)
            fake = finalgen(cords, style, melody, groove)
            # bp()
            preds = fake.cpu().detach().numpy()
            music_data = postProcess(preds)
            filename = 'myexample' + str(i) + '.midi'
            music_data.write('midi', fp=filename)


    ##Now for inspirational generation
    ig = InspirationalGeneration(finalgen, finalcritic)

    ig.inspirational_generation("output_midi/myexample0.midi")