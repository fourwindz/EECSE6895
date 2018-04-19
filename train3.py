from __future__ import print_function

import torch

import torch.nn as nn

from torch.autograd import Variable

import torch.optim as optim

import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import pandas as pd


class Sequence(nn.Module):

    def __init__(self):

        super(Sequence, self).__init__()

        self.lstm1 = nn.LSTMCell(1, 51)

        self.lstm2 = nn.LSTMCell(51, 51)

        self.linear = nn.Linear(51, 1)



    def forward(self, input, future = 0):

        outputs = []

        h_t = Variable(torch.zeros(input.size(0), 51).double(), requires_grad=False)

        c_t = Variable(torch.zeros(input.size(0), 51).double(), requires_grad=False)

        h_t2 = Variable(torch.zeros(input.size(0), 51).double(), requires_grad=False)

        c_t2 = Variable(torch.zeros(input.size(0), 51).double(), requires_grad=False)



        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):

            h_t, c_t = self.lstm1(input_t, (h_t, c_t))

            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))

            output = self.linear(h_t2)

            outputs += [output]

        for i in range(future):# if we should predict the future

            h_t, c_t = self.lstm1(output, (h_t, c_t))

            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))

            output = self.linear(h_t2)

            outputs += [output]

        outputs = torch.stack(outputs, 1).squeeze(2)

        return outputs


if __name__ == '__main__':

    #read exchange rate---------------------------------------
    uscdn = pd.read_csv('uscdn.csv')

    import copy
    rate = copy.copy(uscdn['Close'])
   
    #how many days to forecast
    future = 1250

    #retrain for each forecast period
    forecast_total = 2500
    passes = int(forecast_total/future)
    for j in range(passes):
        print (j)
        
        pass_end = int(2500 + j*(forecast_total/passes))
        print (pass_end)
             
        #train with rolling window 15 days back for forecast start
        ratedata = rate[pass_end-250:pass_end]
        for i in range(15):
            ratedata = np.vstack((ratedata,rate[pass_end-250-i:pass_end-i]))    
        ratedata = ratedata.T

        torch.save(ratedata, open('traindata.pt', 'wb'))
    
    
        # set random seed to 0
    
        np.random.seed(0)
    
        torch.manual_seed(0)
    
        # load data and make training set
    
        data = torch.load('traindata.pt')
    
        input = Variable(torch.from_numpy(data[3:, :-1]), requires_grad=False)
    
        target = Variable(torch.from_numpy(data[3:, 1:]), requires_grad=False)
    
        test_input = Variable(torch.from_numpy(data[:1, :-1]), requires_grad=False)
    
        test_target = Variable(torch.from_numpy(data[:1, 1:]), requires_grad=False)
    
        # build the model
    
        seq = Sequence()
    
        seq.double()
    
        criterion = nn.MSELoss()
    
        # use LBFGS as optimizer since we can load the whole data to train
    
        optimizer = optim.LBFGS(seq.parameters(), lr=0.8)
        
        #begin to train
    
        for i in range(1):
    
            print('STEP: ', i)
    
            def closure():
    
                optimizer.zero_grad()
    
                out = seq(input)
    
                loss = criterion(out, target)
    
                print('loss:', loss.data.numpy()[0])
    
                loss.backward()
    
                return loss
    
            optimizer.step(closure)
    
            # begin to predict
    
            pred = seq(test_input, future = future)
    
            loss = criterion(pred[:, :-future], test_target)
    
            print('test loss:', loss.data.numpy()[0])
    
            y = pred.data.numpy()
    
            # draw the result
    
            plt.figure(figsize=(30,10))
    
            plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
    
            plt.xlabel('x', fontsize=20)
    
            plt.ylabel('y', fontsize=20)
    
            plt.xticks(fontsize=20)
    
            plt.yticks(fontsize=20)
    
            def draw(yi):
    
                #plt.plot(np.arange(input.size(1)), yi[:input.size(1)], color, linewidth = 2.0)
                plt.plot(np.arange(rate.shape[0]), rate, 'r', linewidth = 2.0)
    
                #plt.plot(np.arange(input.size(1), input.size(1) + future), yi[input.size(1):], color + ':', linewidth = 2.0)
                plt.plot(np.arange(2500, 2500 + future), yi[input.size(1):], 'g' + ':', linewidth = 2.0)
    
            draw(y[0])
    
            plt.savefig('predict%d.pdf'%i)
    
            plt.close()