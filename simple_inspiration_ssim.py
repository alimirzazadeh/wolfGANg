# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import json
from nevergrad.optimization import optimizerlib
from copy import deepcopy

from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ipdb import set_trace as bp
import libs.pytorch_ssim as pytorch_ssim
from libs.midi2numpy import midiToNumpy
from torch.autograd import Function

class InspirationalGeneration():

    def __init__(self, generator, critic):
        self.generator = generator
        self.critic = critic
        for param in self.generator.parameters():
            param.requires_grad = False
        for param in self.critic.parameters():
            param.requires_grad = False
        self.device = 'cuda:0'
        self.ssim = pytorch_ssim.SSIM()
        self.lossTracker = []

    def buildFeatureExtractor(self, pathModel, resetGrad=True):
        modelData = torch.load(pathModel)
        fullDump = modelData.get("fullDump", False)
        if fullDump:
            model = modelData['model']
        else:
            modelType = loadmodule(
                modelData['package'], modelData['network'], prefix='')
            model = modelType(**modelData['kwargs'])
            model = cutModelHead(model)
            model.load_state_dict(modelData['data'])

        for param in model.parameters():
            param.requires_grad = resetGrad

        mean = modelData['mean']
        std = modelData['std']

        return model

    def reshaper(self, arr):
        return arr.unsqueeze(0).unsqueeze(0).float()

    def splitInputToParts(self, noise_input, batch_size):
        # noise_input = noise_input.view(-1)
        cords = noise_input[:,:1 * 32].reshape((batch_size, 32))
        style = noise_input[:,1 * 32 :2 * 32].reshape((batch_size, 32))
        #bp()
        melody = noise_input[:,2 * 32:6 * 32].reshape((batch_size, 4, 32))
        groove = noise_input[:,6 * 32:10 * 32].reshape((batch_size, 4, 32))
        return cords, style, melody, groove


    def midiWavTransform(self, midi_path):
        self.fs.midi_to_audio(midi_path, "temp.wav")
        return "temp.wav"
        # samplerate, data = wavfile.read('temp.wav')
        # return data


    def encoder(self, output):
        # aa = torch.tensor(output, requires_grad=True)
        m = nn.Softmax(dim=-1)
        ae = m(output * 10000)

        # ab = torch.max(output, axis=-1, keepdim=True)
        # ab = ab.values.repeat(1,1,1,1,output.shape[-1])
        # ac = output - ab
        # ae = torch.floor(ac) + 1
        bb = ae.view(ae.shape[1], ae.shape[2]*ae.shape[3], ae.shape[4])
        fin = bb.view(bb.shape[0] * bb.shape[1], bb.shape[2])
        # # bp()
        # m = nn.ReLU()
        # fin = m(fin)
        return fin

    def saveAsMIDI(self, output, filepath):
        preds = output.cpu().detach().numpy()
        music_data = postProcess(preds)
        music_data.write('midi', fp=filepath)

    def pil_loader(self, path):

        # open path as file to avoid ResourceWarning
        # (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')


    def getFeatireSize(self,x):

        s = x.size()
        out = 1
        for p in s[1:]:
            out *= p

        return out


    def gradientDescentOnInput(self,
                               input,
                               featureExtractors,
                               imageTransforms,
                               weights=None,
                               lambdaD=0.03,
                               nSteps=600,
                               randomSearch=False,
                               nevergrad=None,
                               lr=1,
                               outPathSave=None):
        r"""
        Performs a similarity search with gradient descent.
        Args:
            model (BaseGAN): trained GAN model to use
            input (tensor): inspiration images for the gradient descent. It should
                            be a [NxCxWxH] tensor with N the number of image, C the
                            number of color channels (typically 3), W the image
                            width and H the image height
            featureExtractors (nn.module): list of networks used to extract features
                                           from an image
            weights (list of float): if not None, weight to give to each feature
                                     extractor in the loss criterion
            visualizer (visualizer): if not None, visualizer to use to plot
                                     intermediate results
            lambdaD (float): weight of the realism loss
            nSteps (int): number of steps to perform
            randomSearch (bool): if true, replace tha gradient descent by a random
                                 search
            nevergrad (string): must be in None or in ['CMA', 'DE', 'PSO',
                                'TwoPointsDE', 'PortfolioDiscreteOnePlusOne',
                                'DiscreteOnePlusOne', 'OnePlusOne']
            outPathSave (string): if not None, path to save the intermediate
                                  iterations of the gradient descent
        Returns
            output, optimalVector, optimalLoss
            output (tensor): output images
            optimalVector (tensor): latent vectors corresponding to the output
                                    images
        """

        if nevergrad not in [None, 'CMA', 'DE', 'PSO',
                             'TwoPointsDE', 'PortfolioDiscreteOnePlusOne',
                             'DiscreteOnePlusOne', 'OnePlusOne']:
            raise ValueError("Invalid nevergard mode " + str(nevergrad))
        randomSearch = randomSearch or (nevergrad is not None)
        print("Running for %d setps" % nSteps)

        # Detect categories
        # bp()
        batch_size = input.size(0)
        varNoise = torch.randn((batch_size,
                                10 * 32),
                               requires_grad=True, device=self.device)

        optimNoise = optim.Adam([varNoise],
                                betas=[0., 0.99], lr=lr)

        # noiseOut = model.test(varNoise, getAvG=True, toCPU=False)
        cords, style, melody, groove = self.splitInputToParts(varNoise, batch_size)
        noiseOut = self.generator(cords, style, melody, groove)
        self.saveAsMIDI(noiseOut,"/home/users/alimirz1/pre_inspiredOutput_001.midi")
        # if not isinstance(featureExtractors, list):
        #     featureExtractors = [featureExtractors]
        if not isinstance(imageTransforms, list):
            imageTransforms = [imageTransforms]

        nExtractors = 1

        if weights is None:
            weights = [1.0 for i in range(nExtractors)]

        if len(imageTransforms) != nExtractors:
            raise ValueError(
                "The number of image transforms should match the number of \
                feature extractors")
        if len(weights) != nExtractors:
            raise ValueError(
                "The number of weights should match the number of feature\
                 extractors")

        featuresIn = []
        for i in range(nExtractors):

        #     if len(featureExtractors[i]._modules) > 0:
        #         featureExtractors[i] = nn.DataParallel(
        #             featureExtractors[i]).train().to(model.device)

        #     featureExtractors[i].eval()
        #     imageTransforms[i] = nn.DataParallel(
        #         imageTransforms[i]).to(model.device)

        #     featuresIn.append(featureExtractors[i](
        #         imageTransforms[i](input.to(model.device))).detach())

        #     if nevergrad is None:
        #         featureExtractors[i].train()
            # bp()
            featuresIn.append(self.reshaper(imageTransforms[i](input.to(self.device))))
            featuresIn[i].requires_grad = False

        lr = 1

        optimalVector = None
        optimalLoss = None

        epochStep = int(nSteps / 3)
        gradientDecay = 0.1

        nImages = input.size(0)
        print(f"Generating {nImages} images")
        if nevergrad is not None:
            optimizers = []
            for i in range(nImages):
                optimizers += [optimizerlib.registry[nevergrad](
                    dimension=320,
                    budget=nSteps)]

        def resetVar(newVal):
            newVal.requires_grad = True
            print("Updating the optimizer with learning rate : %f" % lr)
            varNoise = newVal
            optimNoise = optim.Adam([varNoise],
                                    betas=[0., 0.99], lr=lr)

        # String's format for loss output
        formatCommand = ' '.join(['{:>4}' for x in range(nImages)])
        for iter in range(nSteps):
            # bp()

            optimNoise.zero_grad()
            self.generator.zero_grad()
            self.critic.zero_grad()

            # if randomSearch:
            #     varNoise = torch.randn((nImages,
            #                             320),
            #                            device=self.device)
            #     if nevergrad:
            #         inps = []
            #         for i in range(nImages):
            #             inps += [optimizers[i].ask()]
            #             npinps = np.array(inps)

            #         varNoise = torch.tensor(
            #             npinps, dtype=torch.float32, device=self.device)
            #         varNoise.requires_grad = True
            #         varNoise.to(self.device)

            cords, style, melody, groove = self.splitInputToParts(varNoise, batch_size)
            noiseOut = self.generator(cords, style, melody, groove)
            # sumLoss = torch.zeros(nImages, device=self.device)
            # sumLoss.requires_grad = True

            # loss = (((varNoise**2).mean(dim=1) - 1)**2)
            # sumLoss += loss.view(nImages)
            # loss.sum(dim=0).backward(retain_graph=True)

            for i in range(nExtractors):
                featureOut = self.reshaper(imageTransforms[i](noiseOut))
                # bp()
                diff = 1 - (self.ssim(featuresIn[i], featureOut))
                # bp()
                loss = weights[i] * diff
                sumLoss = loss

                if not randomSearch:
                    retainGraph = (lambdaD > 0) or (i != nExtractors - 1)
                    # bp()
                    # output = torch.autograd.grad(loss,varNoise,create_graph=True)
                    loss.sum().backward(retain_graph=retainGraph)

            # if lambdaD > 0:

            #     loss = -lambdaD * self.critic(noiseOut)[:, 0]
            #     sumLoss += loss

            #     if not randomSearch:
            #         loss.sum(dim=0).backward()
            # bp()
            try:
                sumGrad = torch.sum(varNoise.grad)
            except:
                sumGrad = 0
            self.lossTracker.append(sumLoss.cpu().detach().numpy())
            print("Total Sum: ", sumLoss, " with gradient: ", sumGrad)
            if nevergrad:
                for i in range(nImages):
                    optimizers[i].tell(inps[i], float(sumLoss[i]))
            elif not randomSearch:
                # bp()
                optimNoise.step()

            if optimalLoss is None:
                optimalVector = deepcopy(varNoise)
                optimalLoss = sumLoss

            else:
                optimalVector = torch.where(sumLoss.view(-1, 1) < optimalLoss.view(-1, 1),
                                            varNoise, optimalVector).detach()
                optimalLoss = torch.where(sumLoss < optimalLoss,
                                          sumLoss, optimalLoss).detach()



            if iter % epochStep == (epochStep - 1):
                lr *= gradientDecay
                resetVar(optimalVector)
        cords, style, melody, groove = self.splitInputToParts(optimalVector, batch_size)
        output = self.generator(cords, style, melody, groove).detach()
        self.saveAsMIDI(output,"/home/users/alimirz1/inspiredOutput_001.midi")
        # output = model.test(optimalVector, getAvG=True, toCPU=True).detach()

        # bp()
        print("optimal losses : " + formatCommand.format(
            *"{:10.6f}".format(optimalLoss.item())))
        return output, optimalVector, optimalLoss


    def inspirational_generation(self, midiFile):
        featureExtractors =  None #pytorch_ssim.SSIM()
        imgTransforms = self.encoder
        # bp()
        fullInputs = torch.tensor(midiToNumpy(midiFile)) #_____(midiFile)
        img, outVectors, loss = self.gradientDescentOnInput(fullInputs,
                                                       featureExtractors,
                                                       imgTransforms,
                                                       randomSearch=False,
                                                       nevergrad=None)
        # bp()
        import matplotlib.pyplot as plt
        plt.plot(self.lossTracker)
        plt.savefig("/home/users/alimirz1/lossTracker.png")
        bp()
        pathVectors = basePath + "vector.pt"
        torch.save(outVectors, open(pathVectors, 'wb'))

        # path = basePath + ".jpg"
        # visualisation.saveTensor(img, (img.size(2), img.size(3)), path)
        # outDictData[os.path.splitext(os.path.basename(path))[0]] = \
        #     [x.item() for x in loss]

        # outVectors = outVectors.view(outVectors.size(0), -1)
        # outVectors *= torch.rsqrt((outVectors**2).mean(dim=1, keepdim=True))

        # barycenter = outVectors.mean(dim=0)
        # barycenter *= torch.rsqrt((barycenter**2).mean())
        # meanAngles = (outVectors * barycenter).mean(dim=1)
        # meanDist = torch.sqrt(((barycenter-outVectors)**2).mean(dim=1)).mean(dim=0)
        # outDictData["Barycenter"] = {"meanDist": meanDist.item(),
        #                              "stdAngles": meanAngles.std().item(),
        #                              "meanAngles": meanAngles.mean().item()}

        # path = basePath + "_data.json"
        # outDictData["kwargs"] = kwargs

        # with open(path, 'w') as file:
        #     json.dump(outDictData, file, indent=2)

        # pathVectors = basePath + "vectors.pt"
        # torch.save(outVectors, open(pathVectors, 'wb'))