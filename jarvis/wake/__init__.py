"""Wake-word detection for JARVIS.

``detector``  :class:`jarvis.wake.detector.WakeWord` — openWakeWord with the bundled
``hey_jarvis`` model over the ONNX runtime.

The detector listens to the same float32 16 kHz frames the microphone produces and
reports the single moment JARVIS is addressed. ``openwakeword`` and ``onnxruntime``
are imported inside the detector, never at package import time, so importing this
package succeeds on a machine where the wake-word stack is not installed — the
assistant then simply runs without a wake word instead of failing to start.
"""

from __future__ import annotations

__all__: list[str] = []
