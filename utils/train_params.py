from warnings import warn
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from tensorflow.keras.optimizers import Optimizer, Adam, SGD
from tensorflow.keras.losses import Loss
from tensorflow.keras.metrics import Metric
from utils.losses import SymmetricMAPELoss, MAPELoss, SymmetricMAPE_Sq_Loss
from utils.metrics import SymmetricMAPE, MAPE

@dataclass
class TrainParams:
    batch_size: Optional[int] = 512
    epochs: Optional[int] = 10
    epoch_sample_size: Optional[int] = 2**12
    lr: float = 1.0e-4
    loss: Loss = SymmetricMAPELoss
    optimizer: Optional[Optimizer] = None

    metrics: Dict[str, Metric] = field(default_factory=lambda:
    {
        'smape': SymmetricMAPE(),
        'mape': MAPE()
    })
    train_stats: Dict[str, List] = field(default_factory=lambda: defaultdict(lambda: []))

    def __post_init__(self):
        self._update()

    def _update(self):
        if self.optimizer is None:
            self.optimizer = Adam(self.lr)
        try:
            if issubclass(self.loss, Loss):
                self.loss = self.loss()
        except:
            pass

    def update_param(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                if isinstance(v, str):
                    setattr(self, k, eval(v))
                else:
                    setattr(self, k, v)

        self._update()
        return self

