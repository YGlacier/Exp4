import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import chainerrl

class QFunction(chainer.Chain):
    def __init__(self,input_size=128,output_size=1,hidden_layer_size=256):
        super(QFunction,self).__init__(
        L1=L.Linear(input_size,hidden_layer_size),
        L2=L.Linear(hidden_layer_size,hidden_layer_size),
        L3=L.Linear(hidden_layer_size,output_size)
        )

    def __call__(self,input):
        h=F.tanh(self.L1(input))
        h=F.tanh(self.L2(h))
        h=self.L3(h)
        return chainerrl.action_value.DiscreteActionValue(h)

    def load_model(self,file_name):
        chainer.serializers.load_npz(file_name,self)

    def save_model(self,file_name):
        chainer.serializers.save_npz(file_name,self)
