import math
import numpy as np
from torch import torch
from manifolds.manifold import Manifold 
from manifolds.sphere import _Sphere
from torch.optim.optimizer import Optimizer, required
from proj import proj

class Sigma_sp(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, eight_decay = 0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=0.0001)
        super(Sigma_sp, self).__init__(params, defaults)
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()     
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grads = p.grad.data
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    #  gradient values
                    state['g_0'] = grads.new().resize_as_(grads).zero_()
                state['step'] += 1 
                
                # Projector               
                '''
                The tangent vectors of a surface can be regarded as vectors 
                in the tangent plane and as direction vectors of the tangent lines.
                '''
                gradst = loss.grad
                # Retr
                '''
                The condition of the paper is norm = 1 for x
                grads = _Sphere.retr(gradst),
                It is known from the calculations that
                grads = gradst
                '''
                grads = gradst
        return loss
class Sigma_sp_b(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, eight_decay = 0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=0.0001)
        super(Sigma_sp_b, self).__init__(params, defaults)
 
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()     
    
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grads = p.grad.data
                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    #  gradient values
                    state['g_0'] = grads.new().resize_as_(grads).zero_()
                state['step'] += 1  
                
                # Projector               
                gradst = grads.grad
                # Retr
                gradsr = gradst
                #bgd
                a = 0.3
                gradbgd =  gradsr - a * loss.gard
                grads = gradbgd   
        return loss
class Acc_sp(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, eight_decay = 0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=0.0001)
        super(Acc_sp, self).__init__(params, defaults)
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()     
            
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                        continue
                grads = p.grad.data
                state = self.state[p] 
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    #  gradient values
                    state['g_0'] = grads.new().resize_as_(grads).zero_()
                state['step'] += 1 
                
                # Projector               
                gradst = grads.grad
                gradst_1 = gradst.grad
                # Accelerated Gradient Descent 
                gradsa = gradst_1 + gradst_1 - gradst
                # Retr
                gradsr = gradsa
                grads = gradsr
                
        return loss

class Dual_sp(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, eight_decay = 0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=0.0001)
        super(Dual_sp, self).__init__(params, defaults)
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()     
            
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grads = p.grad.data
                # Projector 
                gradsf = grads.grad
                gradsf = p.grad.data
                # Dual  1.3
                '''
                d_f = p * gradsf - loss,
                max(loss) = 1,min(gradsfm)= p * gradsf - 1
                '''
                df = p * gradsf - 1
                df = p.grad.data
                gradsdf = df.grad
                gradsdf = p.grad.data
                ## Retr
                gradsr = gradsdf
                gradsr = p.grad.data
                grads = gradsr

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    #  gradient values
                    state['g_0'] = grads.new().resize_as_(grads).zero_()
                state['step'] += 1                       
        return loss 
class Dual_A_sp(Optimizer):
    def __init__(self, params,lr=1e-3, betas=(0.9, 0.99), eps=1e-8, eight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps,weight_decay=0.0001)
        super(Dual_A_sp, self).__init__(params, defaults)
       
        
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()     

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grads = p.grad.data
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    #  gradient values
                    state['g_0'] = grads.new().resize_as_(grads).zero_()
                state['step'] += 1
                # Projector 
                gradsf = loss.grad
                gradsf = p.grad.data
                # Dual  1.3+1.7
                '''
                d_f = p * gradsf - loss,
                max(loss) = 1,min(gradsfm)= p * gradsf - 1
                '''
                df = p * gradsf - 1
                df = p.grad.data
                gradsdf = df.grad
                gradsdf = p.grad.data
                ## Retr
                gradsr = gradsdf
                gradsr = p.grad.data
                grads = gradsr
                ##
                if group['weight_decay'] != 0:
                    grads = grads.add(group['weight_decay'], p.data)
                #return grads
                # Decay the first and second moment running average coefficient
                lr = 1e-3
                eps = 1e-8
                beta_1 = 0.99
                beta_2 = 0.99
                grads = beta_1*grads +(1-beta_1)*grads
                grads2 = p.grad.data
                grads2 = beta_2*grads*grads + (1-beta_2)*grads*grads
                grads_ = grads/(1-1-beta_1)
                grads2_ = grads2/(1-1-beta_2)
                grads = grads - lr/(torch.sqrt(grads2_)+eps)*grads_

        return loss   
    
    
