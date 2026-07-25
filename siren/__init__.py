import warnings

warnings.filterwarnings("ignore", module="nemo")
warnings.filterwarnings("ignore", message=".*torchaudio.*")

from .server import app as app  # noqa: E402
