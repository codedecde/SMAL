from __future__ import absolute_import

from allennlp.common import Registrable
from allennlp.training.optimizers import Optimizer

try:
    # Adding support for AdaBound. At a high level, starts off as Adam,
    # and slowly anneals to SGD. See (https://openreview.net/forum?id=Bkg3g2R9FX)
    # for more information.
    import adabound

    Registrable._registry[Optimizer]["adabound"] = adabound.AdaBound
except ImportError:
    print("Use pip install adabound to install adabound")
    pass
