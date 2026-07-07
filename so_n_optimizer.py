import math
import numpy as np
from torch import torch
from manifolds.manifold import Manifold 
from torch.optim.optimizer import Optimizer, required

class Sigma_so(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, eight_decay = 0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=0.0001)
        super(Sigma_so, self).__init__(params, defaults)
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
                def proj(self, X, H):
                    return multiskew(multiprod(multitransp(X), H))
                '''            
                gradst = grads*grads
                # Retr
                grads = gradst
        return loss

class Dual_so_2(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, eight_decay = 0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=0.0001)
        super(Dual_so_2, self).__init__(params, defaults)
    
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
                def proj(self, X, H):
                    return multiskew(multiprod(multitransp(X), H))
                '''
                gradsf = grads*grads
                gradsf = p.grad.data
                # Dual  1.3+1.7
                '''
                d_f = p * gradsf - loss,
                max(loss) = 1,min(gradsfm)= p * gradsf - 1
                '''
                df = p * gradsf - 0.01
                df = p.grad.data
                gradsdf = df.grad
                gradsdf = p.grad.data
                gradst_2 = gradsdf.grad
                gradst_2 = p.grad.data
                ## Retr
                grads = gradst_2
                
        return loss  
class Dual_so_3(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, eight_decay = 0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=0.0001)
        super(Dual_so_3, self).__init__(params, defaults)
    
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
                def proj(self, X, H):
                    return multiskew(multiprod(multitransp(X), H))
                '''
                gradsf = grads*grads*grads
                gradsf = p.grad.data
                # Dual  1.3+1.7
                '''
                d_f = p * gradsf - loss,
                max(loss) = 1,min(gradsfm)= p * gradsf - 1
                '''
                df = p * gradsf - 0.01
                df = p.grad.data
                gradsdf = df.grad
                gradsdf = p.grad.data
                gradst_3 = gradsdf.grad
                gradst_3 = p.grad.data
                ## Retr
                grads = gradst_3
                
        return loss  
class Dual_stiefel(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, eight_decay = 0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=0.0001)
        super(Dual_stiefel, self).__init__(params, defaults)
    
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
                def proj(self, X, H):
                    return multiskew(multiprod(multitransp(X), H))
                '''
                gradsf = grads-grads*grads
                gradsf = p.grad.data
                # Dual  1.3+1.7
                '''
                d_f = p * gradsf - loss,
                max(loss) = 1,min(gradsfm)= p * gradsf - 1
                '''
                df = p * gradsf - 0.01
                df = p.grad.data
                gradsdf = df.grad
                gradsdf = p.grad.data
                gradst_3 = gradsdf.grad
                gradst_3 = p.grad.data
                ## Retr
                grads = gradst_3
                
        return loss 
class Dual_grassmann(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, eight_decay = 0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=0.0001)
        super(Dual_grassmann, self).__init__(params, defaults)
    
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
                def proj(self, X, H):
                    return multiskew(multiprod(multitransp(X), H))
                '''
                gradsf = grads-grads*grads*grads
                gradsf = p.grad.data
                # Dual  1.3+1.7
                '''
                d_f = p * gradsf - loss,
                max(loss) = 1,min(gradsfm)= p * gradsf - 1
                '''
                df = p * gradsf - 0.01
                df = p.grad.data
                gradsdf = df.grad
                gradsdf = p.grad.data
                gradst_3 = gradsdf.grad
                gradst_3 = p.grad.data
                ## Retr
                grads = gradst_3
                
        return loss 


