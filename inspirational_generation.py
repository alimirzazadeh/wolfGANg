import os
import json
from nevergrad.optimization import optimizerlib
from copy import deepcopy

# from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from midi2audio import FluidSynth
from scipy.io import wavfile


class InspirationalGeneration():
    def __init__(self, generator, critic):
        self.generator = generator
        self.critic = critic
        self.featureExtractor = torch.hub.load('harritaylor/torchvggish', 'vggish')
        self.featureExtractor.eval()
        self.fs = FluidSynth()


    def midiWavTransform(self, midi_path):
        self.fs.midi_to_audio(midi_path, "temp.wav")
        return "temp.wav"
        # samplerate, data = wavfile.read('temp.wav')
        # return data


    def encoder(self, wavfile):
        return self.featureExtractor.forward(wavfile)

    def splitInputToParts(self, noise_input, batch_size):
        cords = noise_input[:,:1 * 32]
        style = noise_input[:,1 * 32 :2 * 32]
        melody = noise_input[:,2 * 32:6 * 32].reshape((batch_size, 4, 32))
        groove = noise_input[:,6 * 32:10 * 32].reshape((batch_size, 4, 32))
        return cords, style, melody, groove

    def inspirational_generation(self, ref_midi, batch_size=8, n_steps=100, lambdaD=0.03):
        device = 'cuda:0'
        optimalLoss = None
        optimalVector = None
        ## noise input
        nImages = batch_size
        noise_input = torch.randn(batch_size, 10 * 32).to(device)
        cords, style, melody, groove = splitInputToParts(noise_input, batch_size)

        ## noise output
        noise_out = self.generator(cords, style, melody, groove).detach()

        reference_embedding = encoder(midiWavTransform(ref_midi))

        optimizers = []
        for i in range(nImages):
            optimizers += [optimizerlib.registry['DiscreteOnePlusOne'](parameterization=10*32, budget=n_steps)]


        for step in range(n_steps):
            cords, style, melody, groove = splitInputToParts(noise_input, batch_size)
            noiseOut = self.generator(cords, style, melody, groove).detach()
            cords.requires_grad = True
            style.requires_grad = True
            melody.requires_grad = True
            groove.requires_grad = True

            inps = []
            for i in range(batch_size):
                inps = [optimizers[i].ask()]
                npinps = np.array(inps)

            noise_input = torch.tensor(
                npinps, dtype=torch.float32, device=model.device)
            noise_input.requires_grad = True
            noise_input.to(model.device)
            cords, style, melody, groove = splitInputToParts(noise_input, batch_size)
            noise_out = self.generator(cords, style, melody, groove).detach()

            sumLoss = torch.zeros(nImages, device=model.device)


            #loss 1
            loss = (((noise_input**2).mean(dim=1) - 1)**2)
            sumLoss += loss.view(nImages)
            loss.sum(dim=0).backward(retain_graph=True)

            #loss 2
            featureOut = encoder(midiWavTransform(noise_out))
            diff = ((reference_embedding - featureOut)**2)
            loss = weights * diff.mean(dim=1)
            sumLoss += loss

            #loss 3
            loss = -lambdaD * self.critic(noise_out)
            sumLoss += loss

            for i in range(nImages):
                optimizers[i].tell(inps[i], float(sumLoss[i]))

            if optimalLoss is None:
                optimalVector = deepcopy(noise_input)
                optimalLoss = sumLoss
            else:
                optimalVector = torch.where(sumLoss.view(-1, 1) < optimalLoss.view(-1, 1),
                                            noise_input, optimalVector).detach()
                optimalLoss = torch.where(sumLoss < optimalLoss,
                                          sumLoss, optimalLoss).detach()

        # output = model.test(optimalVector, getAvG=True, toCPU=True).detach()
        cords, style, melody, groove = splitInputToParts(optimalVector, batch_size)
        output = self.generator(cords, style, melody, groove).detach()
        return output