from .gameplay import get_gameplay_service_dep
from .identity import get_identity_service_dep
from .inference import get_inference_service_dep
from .ranking import get_ranking_service_dep

__all__ = [
    "get_gameplay_service_dep",
    "get_identity_service_dep",
    "get_inference_service_dep",
    "get_ranking_service_dep",
]
