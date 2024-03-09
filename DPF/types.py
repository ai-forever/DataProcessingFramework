from typing import Any, Dict, Union

from DPF.modalities import ModalityName

# for data with different modalities
ModalityToDataMapping = Dict[ModalityName, Union[bytes, Any]]
