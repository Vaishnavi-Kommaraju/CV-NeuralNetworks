import numpy as np

def multiclassLRTrain(x, y, param):

    classLabels = np.unique(y)
    numClass = classLabels.shape[0]
    numFeats = x.shape[0]
    numData = x.shape[1]
    print(x.shape, y.shape)

    # Initialize weights randomly (Implement gradient descent)
    model = {}
    model['w'] = np.random.randn(numClass, numFeats)*0.01
    model['b'] = np.ones((numClass,1))
    model['classLabels'] = classLabels
    
    for i in range(param['maxiter']):
        scores = np.dot(model['w'], x) + model['b']
        expScores = np.exp(scores - np.max(scores, axis=0))
        yhat = expScores/np.sum(expScores, axis=0)
        
        #logL = -np.log(yhat[y,range(numData)])
        logL = -(scores[y,range(numData)] - np.max(scores, axis = 0)) + np.log(np.sum(expScores, axis=0))
        #print(logL)
        #print(logL.shape)
        reg = model['w']*model['w']
        CEloss = np.sum(logL)/numData + 0.5*param['lambda']*np.sum(reg)
        
        #Gradient computation
        yhat[y,range(numData)] -= 1
        yhat = yhat/numData
        
        dW = np.dot(yhat, x.T) + param['lambda']*model['w']
        dB = np.sum(yhat, axis=1).reshape(-1,1)
        
        model['w'] -= param['eta']*dW
        model['b'] -= param['eta']*dB
        #if i%50 == 0:
            #print("Iter %d: loss %f" % (i, CEloss))
    return model
