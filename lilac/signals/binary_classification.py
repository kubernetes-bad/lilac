from typing import ClassVar, Iterable, Optional

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing_extensions import override
from pydantic import Field as PydanticField

from ..signal import TextSignal
from ..schema import SignalInputType, field, RichData


class BinaryClassification(TextSignal):
    name: ClassVar[str] = 'binary_classification'
    display_name: ClassVar[str] = 'Binary Classification'
    input_type: ClassVar[SignalInputType] = SignalInputType.TEXT
    local_parallelism: ClassVar[int] = 1
    _tokenizer: AutoTokenizer | None = None
    _model: AutoModelForSequenceClassification | None = None
    _device: torch.device | None = None

    model_id: str = PydanticField(description='Hugging Face Hub ID of the model to run')
    batch_size: int = PydanticField(default=64, description='Batch size of the model. Higher value uses more memory, but faster')

    @override
    def fields(self):
        return field(fields={
            'label_0': 'float32',
            'label_1': 'float32',
        })

    @override
    def setup(self):
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        self._device = BinaryClassification.get_device()
        self._model.to(self._device)

    @override
    def teardown(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None

        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._device = None
        super().teardown()

    @override
    def compute(self, data: Iterable[RichData]) -> Iterable[Optional[dict]]:
        batch = []

        for text in data:
            batch.append(text)
            if len(batch) >= self.batch_size:
                yield from self.process_batch(batch)
                batch = []

        if batch:
            yield from self.process_batch(batch)

    def process_batch(self, batch: list[str]) -> Iterable[Optional[dict]]:
        inputs = self._tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs)

        probabilities = torch.softmax(outputs.logits, dim=1)

        for probs in probabilities:
            yield {
                'label_0': probs[0].item(),
                'label_1': probs[1].item(),
            }

    @staticmethod
    def get_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
