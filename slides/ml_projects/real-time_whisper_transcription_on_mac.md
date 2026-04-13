# Real-time Whisper Transcription on macOS

Capture system audio output and transcribe it live using OpenAI Whisper.
Since macOS doesn't allow direct speaker tapping, a virtual loopback device mirrors audio into Python.

## Architecture

```
macOS system audio → BlackHole (loopback) → Python / PyAudio chunks → Whisper → subtitles printed live
```

---

## Step 1 — Install BlackHole

BlackHole is a free, open-source virtual audio loopback driver. It is actively maintained (18k+ GitHub stars, last updated Dec 2025) and works on both Intel and Apple Silicon without kernel extensions.

> **Note:** The old site `existingaudio.com` is dead. Use the real source below.

```bash
brew install blackhole-2ch
```

Or download the installer directly from:
**https://github.com/ExistentialAudio/BlackHole**

---

## Step 2 — Create a Multi-Output Device

This lets audio play normally through your speakers *and* get mirrored into BlackHole simultaneously.

1. Open **Audio MIDI Setup** (Spotlight → "Audio MIDI Setup")
2. Click **+** → *Create Multi-Output Device*
3. Check both **BlackHole 2ch** and your real speakers/headphones
4. Go to **System Settings → Sound** and set this Multi-Output Device as your output

---

## Step 3 — Install Python Dependencies

```bash
pip install openai-whisper pyaudio numpy
brew install ffmpeg
```

---

## Step 4 — The Script

```python
import whisper
import pyaudio
import numpy as np
import threading
import queue
import time

# ── Config ────────────────────────────────────────────────
DEVICE_NAME   = "BlackHole 2ch"   # must match exactly
SAMPLE_RATE   = 16000
CHUNK_SECONDS = 5                  # transcribe every N seconds
MODEL_SIZE    = "base"             # tiny / base / small / medium / large
# ──────────────────────────────────────────────────────────

CHUNK_FRAMES = SAMPLE_RATE * CHUNK_SECONDS
audio_queue = queue.Queue()


def find_device_index(p, name):
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if name.lower() in info["name"].lower() and info["maxInputChannels"] > 0:
            return i
    return None


def recorder(p, device_index):
    """Continuously read audio from BlackHole into the queue."""
    stream = p.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=1024,
    )
    print(f"[recording from '{DEVICE_NAME}' — speak or play audio...]")
    buffer = []
    while True:
        data = stream.read(1024, exception_on_overflow=False)
        chunk = np.frombuffer(data, dtype=np.float32)
        buffer.append(chunk)
        if sum(len(c) for c in buffer) >= CHUNK_FRAMES:
            audio_queue.put(np.concatenate(buffer))
            buffer = []


def transcriber(model):
    """Pull chunks from the queue and transcribe them."""
    while True:
        audio = audio_queue.get()
        if audio is None:
            break
        result = model.transcribe(audio, fp16=False, language=None)
        text = result["text"].strip()
        if text:
            print(f"\n[{time.strftime('%H:%M:%S')}]  {text}")


def main():
    print(f"Loading Whisper model '{MODEL_SIZE}'...")
    model = whisper.load_model(MODEL_SIZE)

    p = pyaudio.PyAudio()
    device_index = find_device_index(p, DEVICE_NAME)

    if device_index is None:
        print(f"ERROR: Could not find input device '{DEVICE_NAME}'.")
        print("Available input devices:")
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                print(f"  [{i}] {info['name']}")
        return

    t_rec  = threading.Thread(target=recorder,    args=(p, device_index), daemon=True)
    t_tran = threading.Thread(target=transcriber, args=(model,),          daemon=True)
    t_rec.start()
    t_tran.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopped.")
        audio_queue.put(None)


if __name__ == "__main__":
    main()
```

---

## Tips

| Setting | Recommendation |
|---|---|
| `MODEL_SIZE = "base"` | Good default — fast on most Macs |
| `MODEL_SIZE = "small"` or `"medium"` | Better accuracy on M-series chips |
| `CHUNK_SECONDS = 3` | Snappier output, less sentence context |
| `CHUNK_SECONDS = 8` | Better sentence context, slightly more lag |

- Whisper **auto-detects language** — works for any language out of the box.
- When done, switch system output back to your speakers directly (remove the Multi-Output Device in Audio MIDI Setup).

---

## BlackHole Alternatives

| Tool | Cost | Notes |
|---|---|---|
| **BlackHole** | Free / open-source | Recommended. Apple Silicon native, zero latency |
| **Soundflower** | Free | Largely unmaintained, not recommended on Sonoma+ |
| **Loopback** (Rogue Amoeba) | ~$99 | Most polished option, visual routing UI |
