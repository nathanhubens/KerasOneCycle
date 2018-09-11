from keras.callbacks import Callback
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt

#####################
##### OneCycle  #####
#####################

          
class OneCycle(Callback):
    """This callback implements a cyclical learning rate and momentum policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration 
    For more detail, please see paper.
    
    # Example
        ```python
            clr = OneCycle(min_lr=1e-3, max_lr=1e-2,
                      min_mtm=0.85, max_mtm=0.95,
                      annealing=0.1,step_size=np.ceil((X_train.shape[0]*epochs/batch_size)))
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    
    # Arguments
        min_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - min_lr).
        step_size: number of training iterations in the cycle. To define as `np.ceil((X_train.shape[0]*epochs/batch_size))`
        max_mtm : initial value of the momentum    
        min_mtm : lower boundary in the cycle.
        annealing_stage : percentage of the iterations where the lr
                    will decrease lower than its min_lr
        annealing_rate : in annealing phase learning rate will be decreased to annealing_rate*min_lr
                    
        # References
        Original paper: https://arxiv.org/pdf/1803.09820.pdf
        Inspired by : https://sgugger.github.io/the-1cycle-policy.html#the-1cycle-policy

    """

    def __init__(self, min_lr=1e-5, max_lr=1e-2, min_mtm = 0.85, max_mtm=0.95, training_iterations=1000.,
                 annealing_stage=0.1, annealing_rate=0.01):

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.min_mtm = min_mtm
        self.max_mtm = max_mtm
        self.annealing_stage = annealing_stage
        self.step_size = training_iterations*(1-self.annealing_stage)/2
        self.min_annealing_lr = annealing_rate * min_lr
        self.iterations = 0.
        self.training_iterations = training_iterations
        self.history = {}
        
    def clr(self):
        if self.iterations < 2*self.step_size :
            x = np.abs(self.iterations/self.step_size - 1)
            return self.min_lr + (self.max_lr-self.min_lr)*(1-x)
        else:
            x = min(1, float(self.iterations - 2 * self.step_size) / (self.training_iterations - 2 * self.step_size))
            return self.min_lr - (self.min_lr - self.min_annealing_lr) * x
        
    
    def cmtm(self):
        if self.iterations < 2*self.step_size :   
            x = np.abs(self.iterations/self.step_size - 1)
        else: 
            x=1
        return self.min_mtm + (self.max_mtm-self.min_mtm)*(x)     
        
    def on_train_begin(self, logs={}):
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.min_lr)
        K.set_value(self.model.optimizer.momentum, self.max_mtm)
         
    def on_batch_end(self, batch, logs=None):
        
        logs = logs or {}
        self.iterations += 1
    
        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('momentum', []).append(K.get_value(self.model.optimizer.momentum))
        self.history.setdefault('iterations', []).append(self.iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
        
        K.set_value(self.model.optimizer.lr, self.clr()) 
        K.set_value(self.model.optimizer.momentum, self.cmtm())
        
    def plot_lr(self):
        plt.xlabel('Training Iterations')
        plt.ylabel('Learning Rate')
        plt.title("CLR - '1 cycle' Policy")
        plt.plot(self.history['iterations'], self.history['lr'])
        
    def plot_mtm(self):
        plt.xlabel('Training Iterations')
        plt.ylabel('Momentum')
        plt.title("CLR - '1 cycle' Policy")
        plt.plot(self.history['iterations'], self.history['momentum'])
        