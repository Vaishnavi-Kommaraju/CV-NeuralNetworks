import numpy as np
import os
import utils
import time
from extractDigitFeatures import extractDigitFeatures
from trainModel import trainModel
from evaluateLabels import evaluateLabels
from evaluateModel import evaluateModel
import argparse
#import wandb

if __name__ == "__main__":
    #wandb.init(project="CV-P5")
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataType', default='digits-normal.mat', type=str) #'digits-normal.mat', 'digits-scaled.mat', 'digits-jitter.mat'
    parser.add_argument('--featureType', default='lbp', type=str) #lbp,pixel,hog
    parser.add_argument('--normalisation', default='sqrt', type=str) #sqrt, l2
    parser.add_argument('--eta', default=0.01, type=float)
    parser.add_argument('--lambda_', default=0.008, type=float)
    parser.add_argument('--iter_', default=1000, type=int)
    args = parser.parse_args()

    for key in ['dataType','featureType', 'normalisation', 'eta', 'lambda_']:
        #wandb.config[key] = args[key]
        print('!')

    trainSet = 1
    valSet = 2
    testSet = 3
    #Load data
    path = os.path.join('..', 'data', args.dataType)
    data = utils.loadmat(path)
    #wandb.log({'info': args})
    print({'info': args})
    # Extract features
    tic = time.time()
    features = extractDigitFeatures(data['x'], args.featureType)
    if args.normalisation == 'sqrt':
        features = np.sqrt(features)
    elif args.normalisation == 'l2':
        features = features/(np.sqrt(np.sum(features**2, axis=0, keepdims=True)))
    #wandb.log({'FeatureExtractionTime': time.time()-tic})
    print({'FeatureExtractionTime': time.time()-tic})
    # Train model
    tic = time.time()
    model = trainModel(features[:, data['set']==trainSet], data['y'][data['set']==trainSet], eta=args.eta, lambda_=args.lambda_, iter_=args.iter_)
    #wandb.log({'TrainTime': time.time()-tic})
    print({'TrainTime': time.time()-tic})
    # Train model accuracy
    ypred = evaluateModel(model, features[:, data['set']==trainSet])
    y = data['y'][data['set']==trainSet]
    (acc, conf) = evaluateLabels(y, ypred, False)
    #wandb.log({'TrainAccuracy': acc*100})
    print({'TrainAccuracy': acc*100})

    # Test set accuracy
    ypred = evaluateModel(model, features[:, data['set']==valSet])
    y = data['y'][data['set']==valSet]
    (acc, conf) = evaluateLabels(y, ypred, False)
    #wandb.log({'ValAccuracy': acc*100})
    print({'ValAccuracy': acc*100})
    
    # Train again
    tic = time.time()
    feature_1, data_1 = features[:,data['set']==trainSet], data['y'][data['set']==trainSet]
    feature_2, data_2 = features[:,data['set']==valSet], data['y'][data['set']==valSet]
    features_ = np.concatenate((feature_1, feature_2),axis=1)
    data_ = np.concatenate((data_1, data_2),axis=0)
    model = trainModel(features_, data_, eta=args.eta, lambda_=args.lambda_, iter_=args.iter_)
    #print('{:.2f}s to train model'.format(time.time()-tic))
    print('Retrain done')

    # Test set accuracy
    ypred = evaluateModel(model, features[:, data['set']==testSet])
    y = data['y'][data['set']==testSet]
    (acc, conf) = evaluateLabels(y, ypred, False)
    #wandb.log({'TestAccuracy': acc*100})
    print({'TestAccuracy': acc*100})
