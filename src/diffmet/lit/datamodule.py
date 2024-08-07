from functools import cached_property
from torch.utils.data import DataLoader
from lightning.pytorch import LightningDataModule
from ..utils.lit import get_class


class DataModule(LightningDataModule):

    def __init__(self,
                 dataset_class_path: str,
                 train_files: list[str],
                 val_files: list[str],
                 test_files: list[str],
                 predict_files: list[str] = [],
                 train_sampler_class_path: str | None = None,
                 batch_size: int = 256,
                 eval_batch_size: int = 512,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

    @cached_property
    def dataset_cls(self):
        return get_class(self.hparams['dataset_class_path'])

    @cached_property
    def train_sampler_cls(self):
        if self.hparams['train_sampler_class_path'] is not None:
            return get_class(self.hparams['train_sampler_class_path'])

    @cached_property
    def train_set(self):
        return self.dataset_cls.load(self.hparams['train_files'])

    @cached_property
    def val_set(self):
        return self.dataset_cls.load(self.hparams['val_files'])

    @cached_property
    def test_set(self):
        return self.dataset_cls.load(self.hparams['test_files'])

    @cached_property
    def predict_set(self):
        return self.dataset_cls.load(self.hparams['predict_files'])

    @cached_property
    def train_sampler(self):
        if self.train_sampler_cls is not None:
            return self.train_sampler_cls(self.train_set)

    def setup(self, stage: str) -> None:
        if stage == 'fit':
            self.train_set
            self.val_set
        elif stage == 'validate':
            self.val_set
        elif stage == 'test':
            self.test_set
        elif stage == 'predict':
            self.predict_set
        else:
            raise ValueError(f'{stage=}')

    def teardown(self, stage: str) -> None:
        if stage == 'fit':
            delattr(self, 'train_set')
            delattr(self, 'val_set')
            delattr(self, 'train_sampler')
        elif stage == 'test':
            delattr(self, 'test_set')
        elif stage == 'predict':
            delattr(self, 'predict_set')


    def train_dataloader(self):
        dataset = self.train_set

        kwargs = {}
        if self.train_sampler is None:
            kwargs['shuffle'] = True
        else:
            kwargs['sampler'] = self.train_sampler

        return DataLoader(
            dataset=dataset,
            batch_size=self.hparams['batch_size'],
            collate_fn=dataset.collate,
            drop_last=True,
            **kwargs
        )

    def _eval_dataloader(self, dataset):
        return DataLoader(
            dataset=dataset,
            batch_size=self.hparams['eval_batch_size'],
            shuffle=False,
            collate_fn=dataset.collate,
            drop_last=False,
        )

    def val_dataloader(self):
        return self._eval_dataloader(self.val_set)

    def test_dataloader(self):
        return self._eval_dataloader(self.test_set)

    def predict_dataloader(self):
        return self._eval_dataloader(self.predict_set)
