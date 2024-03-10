from typing import Any, Union

from DPF.modalities import ModalityName

# for data with different modalities
ModalityToDataMapping = dict[ModalityName, Union[bytes, Any]]
