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
from impdb import set_trace as bp
import libs.pytorch_ssim as pytorch_ssim
from libs.midi2numpy import midiToNumpy

class InspirationalGeneration():

    def __init__(self, generator, critic):
        self.generator = generator
        self.critic = critic
        self.device = 'cuda:0'
        self.ssim = pytorch_ssim.SSIM()

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

    def midiWavTransform(self, midi_path):
        self.fs.midi_to_audio(midi_path, "temp.wav")
        return "temp.wav"
        # samplerate, data = wavfile.read('temp.wav')
        # return data


    def encoder(self, output):
        aa = torch.tensor(output, requires_grad=True)
        ab = torch.max(aa, axis=-1, keepdim=True)
        ab = ab.values.repeat(1,1,1,1,aa.shape[-1])
        ac = aa - ab
        ae = torch.floor(ac) + 1
        bb = ae.view(ae.shape[1], ae.shape[2]*ae.shape[3], ae.shape[4])
        fin = bb.view(bb.shape[1],bb.shape[0] * bb.shape[2])
        bp()
        return fin



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
                               model,
                               input,
                               featureExtractors,
                               imageTransforms,
                               weights=None,
                               lambdaD=0.03,
                               nSteps=6000,
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
        varNoise = torch.randn((input.size(0),
                                model.config.noiseVectorDim +
                                model.config.categoryVectorDim),
                               requires_grad=True, device=self.device)

        optimNoise = optim.Adam([varNoise],
                                betas=[0., 0.99], lr=lr)

        noiseOut = model.test(varNoise, getAvG=True, toCPU=False)

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
            featuresIn.append(self.reshaper(imageTransforms[i](input.to(self.device))))

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
                    dimension=model.config.noiseVectorDim +
                    model.config.categoryVectorDim,
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

            optimNoise.zero_grad()
            model.netG.zero_grad()
            model.netD.zero_grad()

            if randomSearch:
                varNoise = torch.randn((nImages,
                                        model.config.noiseVectorDim +
                                        model.config.categoryVectorDim),
                                       device=self.device)
                if nevergrad:
                    inps = []
                    for i in range(nImages):
                        inps += [optimizers[i].ask()]
                        npinps = np.array(inps)

                    varNoise = torch.tensor(
                        npinps, dtype=torch.float32, device=self.device)
                    varNoise.requires_grad = True
                    varNoise.to(self.device)

            noiseOut = model.netG(varNoise)
            sumLoss = torch.zeros(nImages, device=self.device)

            loss = (((varNoise**2).mean(dim=1) - 1)**2)
            sumLoss += loss.view(nImages)
            loss.sum(dim=0).backward(retain_graph=True)

            for i in range(nExtractors):
                featureOut = self.reshaper(imageTransforms[i](noiseOut))
                diff = 1 - (self.ssim(featuresIn[i], featureOut))
                bp()
                loss = weights[i] * diff
                sumLoss += loss

                if not randomSearch:
                    retainGraph = (lambdaD > 0) or (i != nExtractors - 1)
                    loss.sum(dim=0).backward(retain_graph=retainGraph)

            if lambdaD > 0:

                loss = -lambdaD * model.netD(noiseOut)[:, 0]
                sumLoss += loss

                if not randomSearch:
                    loss.sum(dim=0).backward()

            if nevergrad:
                for i in range(nImages):
                    optimizers[i].tell(inps[i], float(sumLoss[i]))
            elif not randomSearch:
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

        output = model.test(optimalVector, getAvG=True, toCPU=True).detach()


        print("optimal losses : " + formatCommand.format(
            *["{:10.6f}".format(optimalLoss[i].item())
              for i in range(nImages)]))
        return output, optimalVector, optimalLoss


    def inspiration_generation(self, midiFile):
        generator = self.generator
        discriminator = self.critic
        featureExtractors =  None #pytorch_ssim.SSIM()
        imgTransforms = self.encoder
        fullInputs = torch.tensor(midiToNumpy(midiFile)) #_____(midiFile)
        img, outVectors, loss = gradientDescentOnInput(generator, discriminator,
                                                       fullInputs,
                                                       featureExtractors,
                                                       imgTransforms,
                                                       randomSearch=False,
                                                       nevergrad=None,
                                                       outPathSave=outPathDescent)
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