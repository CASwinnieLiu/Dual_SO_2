A Novel Dual Decomposition Optimization  Method on SO(2) Manifold for Deep Neural Networks

## 1 Import Optimizer
When using our method, locate the training file for your deep learning model, and then add the following lines of code:
```
from so_n_optimizer import Dual_so_2

```
```
from sphere_optimizer import Dual_sp

```
## 2  Setup Optimizer
When the optimizer code is 
```
optimizer = optim.AdamW(model.parameters(), lr=args.lr_max, weight_decay=args.weight_decay),

```
You can comment out the line above and change it to
```
optimizer = Dual_so_2(model.parameters(), lr=args.lr_max)
optimizer = optim.Adam(model.parameters(), lr=args.lr_max, weight_decay=args.weight_decay)

```
and
```
optimizer = Dual_sp(model.parameters(), lr=args.lr_max)
optimizer = optim.Adam(model.parameters(), lr=args.lr_max, weight_decay=args.weight_decay)

```
