from typing import Union, Dict, Any

from DPF.modalities import ModalityName

# for data with different modalities
ModalityToDataMapping = Dict[ModalityName, Union[bytes, Any]]