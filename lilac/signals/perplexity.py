"""Perplexity detection of a document."""
from typing import ClassVar, Iterator, Optional

import modal
from typing_extensions import override

from ..schema import Field, SignalInputType, field
from ..signal import TextSignal
from ..utils import chunks


class Perplexity(TextSignal):
  """Calculates the perplexity of a document."""

  name: ClassVar[str] = 'perplexity'
  display_name: ClassVar[str] = 'Perplexity score'

  input_type: ClassVar[SignalInputType] = SignalInputType.TEXT

  local_batch_size: ClassVar[Optional[int]] = -1

  @override
  def fields(self) -> Field:
    return field('float32')

  @override
  def compute(self, data: Iterator[str]) -> Iterator[Optional[float]]:
    pplx = modal.Function.lookup('perplexity', 'Perplexity.compute')

    batches = ({'docs': batch} for batch in chunks(data, 256))
    for response in pplx.map(batches, order_outputs=True):
      yield from response['scores']
