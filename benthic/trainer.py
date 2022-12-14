from dataclasses import dataclass

@dataclass
class ModelTrainer():
    optimizer: object
    criterion: object
    device: object
    batch_size: int
    scheduler: object=None


    def get_optimizer(self) -> object:
        return self.optimizer

    def get_criterion(self) -> object:
        return self.criterion

    def get_device(self) -> object:
        return self.device

    def get_scheduler(self) -> object:
        return self.scheduler

    def get_batch_size(self) -> int:
        return self.batch_size

    def has_scheduler(self) -> bool:
        return self.scheduler is not None

