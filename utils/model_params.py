from dataclasses import dataclass
from typing import Iterable, Tuple, Optional


@dataclass
class BlockType:
    SEASONALITY_BLOCK: str = 'seasonality'
    TREND_BLOCK: str = 'trend'
    GENERIC_BLOCK: str = 'generic'


@dataclass
class NBeatsParams:
    name: str = 'NBeatsTF'
    backcast_length: Optional[int] = None
    forecast_length: Optional[int] = None
    n_stacks: Optional[int] = 2
    n_channels: int = 1

    # interpretable = trend --> seasonality
    # (TREND_BLOCK, SEASONALITY_BLOCK)
    stack_types: Tuple[str, ...] = (BlockType.GENERIC_BLOCK, )
    nb_blocks_per_stack: int = 3
    thetas_dim: Tuple[int, ...] = (16,)
    share_weights_in_stack: bool = False
    hidden_layer_units: Tuple[int, ...] = (512,)

    def __post_init__(self):
        self._update()

    def _update(self):
        if len(self.stack_types) == 1:
            self.stack_types = tuple([self.stack_types[0] for _ in range(self.n_stacks)])
        assert len(self.stack_types) == self.n_stacks
        if len(self.thetas_dim) == 1:
            self.thetas_dim = tuple([self.thetas_dim[0] for _ in range(self.n_stacks)])
        assert len(self.thetas_dim) == len(self.stack_types)
        if len(self.hidden_layer_units) == 1:
            self.hidden_layer_units = (self.hidden_layer_units[0],) * len(self.stack_types)
        assert len(self.hidden_layer_units) == len(self.stack_types)

    def update_param(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                if k == 'name':
                    self.name = v
                elif isinstance(v, str):
                    setattr(self, k, eval(v))
                else:
                    setattr(self, k, v)

        self._update()
        return self

@dataclass
class WNBeatsParams(NBeatsParams):
    name: str = 'WNBeatsTF'
    mix_method: str = 'weighted_dropout'
    dropout: float = 0.5
    mixer_hidden_units: int = 512


@dataclass
class EnsembleParams:
    name: str = 'NBeatsEnsemble'
    aggregation_method: str = 'median' # [median | mean]

    def update_param(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                if k == 'aggregation_method':
                    self.aggregation_method = v
                elif k == 'name':
                    self.name = v
                elif isinstance(v, str):
                    setattr(self, k, eval(v))
                else:
                    setattr(self, k, v)
        return self


@dataclass
class WeightedEnsembleParams:
    name: str = 'NBeatsWeightedEnsemble'
    weighted_aggregation_method: str = 'argmax_dropout' # [argmax_dropout | argmax | dropout | naiive]
    hidden_layer_units: int = 512
    submodel_names: Optional[Iterable[str]] = None

    def update_param(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                if k == 'weighted_aggregation_method':
                    self.weighted_aggregation_method = v
                elif k == 'name':
                    self.name = v
                elif isinstance(v, str):
                    setattr(self, k, eval(v))
                else:
                    setattr(self, k, v)
        return self

@dataclass
class MSCNNParams:
    name: str = 'MSCNN'
    backcast_length: Optional[int] = None
    forecast_length: Optional[int] = None
    n_stacks: Optional[int] = 2
    n_channels: int = 1
    layer_widths: int = 512

    def __post_init__(self):
        self._update()

    def _update(self):
        pass

    def update_param(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                if k == 'name':
                    self.name = v
                elif isinstance(v, str):
                    setattr(self, k, eval(v))
                else:
                    setattr(self, k, v)

        self._update()

        return self