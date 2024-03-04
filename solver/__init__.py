from solver.scheduler import ReduceLROnPlateau, MultiStepLR, ConstantLRWithWarmup, ConstantLRWithCooldownByRound
from .optimizer import Optimizer


def make_scheduler(cfg, optimizer):
    if cfg.SOLVER.NAME == None:
        if len(cfg.SOLVER.STEP) == 0:
            return ReduceLROnPlateau(cfg, optimizer)
        else:
            return MultiStepLR(cfg, optimizer)
    else:
        if cfg.SOLVER.NAME == 'constant':
            return ConstantLRWithWarmup(cfg, optimizer)
        elif cfg.SOLVER.NAME == 'cooldown':
            return ConstantLRWithCooldownByRound(cfg, optimizer)

