"""
=============================================================================
EIT Audio Transcription Pipeline
=============================================================================
Project: Audio-to-Text Transcription for Second/Additional Language Learners
Goal:    Convert learner EIT audio responses to text with ~90% accuracy
         compared to human transcribers.

Pipeline Steps:
  1. Audio Preprocessing   - noise reduction, normalization, segmentation
  2. Transcription         - Whisper-based ASR (fine-tuned for learner speech)
  3. Post-processing       - error correction for common learner patterns
  4. Output                - structured CSV/JSON transcription results
=============================================================================
"""

import os
import wave
import struct
import json
import csv
import re
import time
import logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Setup logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("EIT-Pipeline")


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------
@dataclass
class TranscriptionResult:
    learner_id: str
    sentence_id: str
    audio_file: str
    raw_transcription: str
    corrected_transcription: str
    confidence_score: float
    duration_seconds: float
    processing_time_ms: float
    flags: List[str]  # e.g. ["disfluency_detected", "low_confidence"]


# ---------------------------------------------------------------------------
# STEP 1: Audio Preprocessor
# ---------------------------------------------------------------------------
class AudioPreprocessor:
    """
    Cleans raw learner audio before transcription.
    
    Operations:
      - Load WAV file
      - Normalize amplitude
      - Apply noise gate (removes silence/background noise)
      - Segment out leading/trailing silence
    """

    def __init__(self, sample_rate: int = 16000, silence_threshold: float = 0.01,
                 min_speech_duration: float = 0.3):
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.min_speech_duration = min_speech_duration

    def load_audio(self, filepath: str) -> Tuple[np.ndarray, int]:
        """Load a WAV file and return (samples, sample_rate)."""
        with wave.open(filepath, 'r') as wf:
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        return samples, sr

    def normalize(self, samples: np.ndarray) -> np.ndarray:
        """Normalize audio to peak amplitude of 0.9."""
        peak = np.max(np.abs(samples))
        if peak > 0:
            return samples * (0.9 / peak)
        return samples

    def noise_gate(self, samples: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """
        Simple noise gate: zero out samples below threshold.
        Removes constant low-level background noise.
        """
        t = threshold or self.silence_threshold
        gated = samples.copy()
        gated[np.abs(gated) < t] = 0.0
        return gated

    def trim_silence(self, samples: np.ndarray, frame_ms: int = 25) -> np.ndarray:
        """
        Remove leading and trailing silence using energy-based detection.
        Works in frames of `frame_ms` milliseconds.
        """
        frame_len = int(self.sample_rate * frame_ms / 1000)
        energy = np.array([
            np.sqrt(np.mean(samples[i:i+frame_len]**2))
            for i in range(0, len(samples) - frame_len, frame_len)
        ])
        voiced = energy > self.silence_threshold
        if not np.any(voiced):
            return samples  # No voiced frames found, return as is

        start_frame = np.argmax(voiced)
        end_frame = len(voiced) - np.argmax(voiced[::-1])
        start_sample = max(0, start_frame * frame_len - frame_len)
        end_sample = min(len(samples), end_frame * frame_len + frame_len)
        return samples[start_sample:end_sample]

    def process(self, filepath: str) -> Tuple[np.ndarray, int, float]:
        """
        Full preprocessing pipeline.
        Returns (processed_samples, sample_rate, duration_seconds)
        """
        logger.info(f"  Preprocessing: {Path(filepath).name}")
        samples, sr = self.load_audio(filepath)
        samples = self.normalize(samples)
        samples = self.noise_gate(samples)
        samples = self.trim_silence(samples)
        duration = len(samples) / sr
        logger.info(f"    Duration after trim: {duration:.2f}s")
        return samples, sr, duration


# ---------------------------------------------------------------------------
# STEP 2: ASR Engine (Whisper wrapper)
# ---------------------------------------------------------------------------
class WhisperTranscriber:
    """
    Wraps OpenAI Whisper for Spanish learner speech transcription.
    
    In production: loads a fine-tuned Whisper model.
    In demo mode:  returns realistic mock transcriptions.
    
    To use real Whisper:
        pip install openai-whisper
        model = WhisperTranscriber(model_size="medium", use_demo=False)
    """

    # Reference EIT sentences (used in demo mode)
    EIT_REFERENCE_SENTENCES = [
        "El niño come una manzana roja",
        "La profesora habla con los estudiantes",
        "Nosotros vivimos en una ciudad grande",
        "Ella siempre estudia por la noche",
        "Los pájaros cantan en el jardín",
    ]

    # Simulated learner errors (non-native patterns)
    LEARNER_VARIANTS = [
        "El niño come una manzana rojo",           # gender agreement error
        "La profesora habla con estudiantes",       # missing article
        "Nosotros vivimos en ciudad grande",        # missing article
        "Ella siempre estudia en la noche",         # preposition error
        "Los pájaros cantan en el jardín",          # correct
    ]

    def __init__(self, model_size: str = "medium", language: str = "es",
                 use_demo: bool = True, model_path: Optional[str] = None):
        self.model_size = model_size
        self.language = language
        self.use_demo = use_demo
        self.model = None

        if not use_demo:
            self._load_model(model_path)

    def _load_model(self, model_path: Optional[str] = None):
        """Load Whisper model (real usage)."""
        try:
            import whisper
            if model_path:
                logger.info(f"Loading fine-tuned model from: {model_path}")
                self.model = whisper.load_model(model_path)
            else:
                logger.info(f"Loading Whisper {self.model_size} model...")
                self.model = whisper.load_model(self.model_size)
            logger.info("Model loaded successfully.")
        except ImportError:
            logger.warning("Whisper not installed. Run: pip install openai-whisper")
            logger.warning("Falling back to demo mode.")
            self.use_demo = True

    def transcribe(self, audio_file: str, sentence_index: Optional[int] = None
                   ) -> Tuple[str, float]:
        """
        Transcribe audio file.
        Returns (transcription_text, confidence_score 0-1)
        """
        if self.use_demo:
            return self._demo_transcribe(audio_file, sentence_index)

        # Real Whisper transcription
        result = self.model.transcribe(
            audio_file,
            language=self.language,
            task="transcribe",
            fp16=False,
            verbose=False
        )
        text = result["text"].strip()
        # Whisper doesn't give per-utterance confidence natively;
        # estimate from segment no_speech_prob
        segments = result.get("segments", [])
        if segments:
            avg_no_speech = np.mean([s.get("no_speech_prob", 0.1) for s in segments])
            confidence = 1.0 - avg_no_speech
        else:
            confidence = 0.5
        return text, round(confidence, 3)

    def _demo_transcribe(self, audio_file: str, sentence_index: Optional[int] = None
                         ) -> Tuple[str, float]:
        """
        Demo mode: simulate realistic learner transcription output.
        Randomly applies learner-like errors to reference sentences.
        """
        import random
        rng = np.random.default_rng(hash(audio_file) % (2**32))

        if sentence_index is not None and 0 <= sentence_index < len(self.EIT_REFERENCE_SENTENCES):
            ref = self.EIT_REFERENCE_SENTENCES[sentence_index]
            variant = self.LEARNER_VARIANTS[sentence_index]
        else:
            idx = int(rng.integers(0, len(self.EIT_REFERENCE_SENTENCES)))
            ref = self.EIT_REFERENCE_SENTENCES[idx]
            variant = self.LEARNER_VARIANTS[idx]

        # Randomly choose: correct / learner error / partial repetition
        r = rng.random()
        if r < 0.4:
            text = ref           # Correct repetition
            confidence = round(rng.uniform(0.82, 0.95), 3)
        elif r < 0.75:
            text = variant       # Learner error
            confidence = round(rng.uniform(0.70, 0.85), 3)
        else:
            # Partial: only first half of words
            words = ref.split()
            text = " ".join(words[:max(2, len(words)//2)])
            confidence = round(rng.uniform(0.50, 0.72), 3)

        time.sleep(0.05)  # Simulate processing time
        return text, confidence


# ---------------------------------------------------------------------------
# STEP 3: Post-Processor (Learner Language Error Correction)
# ---------------------------------------------------------------------------
class LearnerLanguagePostProcessor:
    """
    Applies rule-based corrections for predictable transcription errors
    that occur specifically in learner/non-native Spanish speech.
    
    Error types handled:
      - ASR phonological confusions (e.g. b/v, ll/y, c/s/z)
      - Common word boundary errors
      - Disfluency markers (um, eh, uh in Spanish)
      - Partial repetitions / self-corrections
    """

    # Spanish phonological confusion pairs (ASR commonly confuses these)
    PHONOLOGICAL_CORRECTIONS = {
        r'\bvamos\b': 'vamos',
        r'\bbamos\b': 'vamos',       # b/v confusion
        r'\balla\b': 'allá',
        r'\baya\b': 'allá',          # ll/y confusion
        r'\baser\b': 'hacer',
        r'\baser\b': 'hacer',        # h-dropping + s/c
        r'\bserca\b': 'cerca',       # s/c confusion
        r'\bkasa\b': 'casa',
        r'\bcasa\b': 'casa',
        r'\bkomer\b': 'comer',
        r'\besta\b': 'está',         # missing accent marks
        r'\bel\b(?= \w+[aeiou])': 'el',
    }

    # Disfluency patterns common in non-native Spanish
    DISFLUENCY_PATTERNS = [
        r'\b(eh|um|uh|ah|eeh|mmm)\b',  # filler words
        r'\b(\w+)-\s*\1',               # word repetition (stammers)
    ]

    # Learner-specific word substitutions (transfer from English)
    TRANSFER_CORRECTIONS = {
        'the': 'el',
        'and': 'y',
        'is': 'es',
        'of': 'de',
    }

    def __init__(self, apply_accent_restoration: bool = True,
                 remove_disfluencies: bool = True):
        self.apply_accent_restoration = apply_accent_restoration
        self.remove_disfluencies = remove_disfluencies

    def correct(self, text: str) -> Tuple[str, List[str]]:
        """
        Apply all post-processing corrections.
        Returns (corrected_text, list_of_flags)
        """
        flags = []
        corrected = text.strip()

        # 1. Detect and optionally remove disfluencies
        for pattern in self.DISFLUENCY_PATTERNS:
            if re.search(pattern, corrected, re.IGNORECASE):
                flags.append("disfluency_detected")
                if self.remove_disfluencies:
                    corrected = re.sub(pattern, '', corrected, flags=re.IGNORECASE)

        # 2. Apply phonological confusion corrections
        for pattern, replacement in self.PHONOLOGICAL_CORRECTIONS.items():
            corrected = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)

        # 3. Fix L1 transfer errors (English words in Spanish response)
        words = corrected.split()
        new_words = []
        for word in words:
            lower = word.lower().strip('.,!?')
            if lower in self.TRANSFER_CORRECTIONS:
                flags.append(f"L1_transfer: '{lower}'")
                new_words.append(self.TRANSFER_CORRECTIONS[lower])
            else:
                new_words.append(word)
        corrected = ' '.join(new_words)

        # 4. Clean up whitespace
        corrected = re.sub(r'\s+', ' ', corrected).strip()

        # 5. Capitalize first letter
        if corrected:
            corrected = corrected[0].upper() + corrected[1:]

        return corrected, flags

    def compute_agreement(self, hypothesis: str, reference: str) -> float:
        """
        Compute word-level agreement score between transcription and reference.
        Simple token overlap (for 90% agreement target evaluation).
        """
        hyp_words = set(hypothesis.lower().split())
        ref_words = set(reference.lower().split())
        if not ref_words:
            return 0.0
        overlap = hyp_words & ref_words
        precision = len(overlap) / len(hyp_words) if hyp_words else 0
        recall = len(overlap) / len(ref_words)
        if precision + recall == 0:
            return 0.0
        f1 = 2 * precision * recall / (precision + recall)
        return round(f1, 3)


# ---------------------------------------------------------------------------
# STEP 4: Full Pipeline Orchestrator
# ---------------------------------------------------------------------------
class EITPipeline:
    """
    Orchestrates the full EIT transcription pipeline:
        Audio File → Preprocess → Transcribe → Post-process → Save Results
    """

    def __init__(self, output_dir: str = "outputs", use_demo: bool = True,
                 whisper_model: str = "medium", model_path: Optional[str] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.preprocessor = AudioPreprocessor()
        self.transcriber = WhisperTranscriber(
            model_size=whisper_model,
            use_demo=use_demo,
            model_path=model_path
        )
        self.postprocessor = LearnerLanguagePostProcessor()
        self.results: List[TranscriptionResult] = []

    def process_file(self, audio_path: str, learner_id: Optional[str] = None,
                     sentence_index: Optional[int] = None) -> TranscriptionResult:
        """Process a single audio file through the full pipeline."""
        filepath = Path(audio_path)
        start_time = time.time()

        lid = learner_id or filepath.stem.split('_')[0]
        sid = filepath.stem

        logger.info(f"Processing: {filepath.name}")

        # Step 1: Preprocess
        _, _, duration = self.preprocessor.process(str(filepath))

        # Step 2: Transcribe
        raw_text, confidence = self.transcriber.transcribe(
            str(filepath), sentence_index=sentence_index
        )
        logger.info(f"  Raw transcription: '{raw_text}' (conf={confidence})")

        # Step 3: Post-process
        corrected_text, flags = self.postprocessor.correct(raw_text)
        if corrected_text != raw_text:
            logger.info(f"  Corrected:         '{corrected_text}'")
        if flags:
            logger.info(f"  Flags: {flags}")

        # Flag low confidence
        if confidence < 0.65:
            flags.append("low_confidence")

        processing_ms = round((time.time() - start_time) * 1000, 1)

        result = TranscriptionResult(
            learner_id=lid,
            sentence_id=sid,
            audio_file=str(filepath),
            raw_transcription=raw_text,
            corrected_transcription=corrected_text,
            confidence_score=confidence,
            duration_seconds=round(duration, 2),
            processing_time_ms=processing_ms,
            flags=flags
        )
        self.results.append(result)
        return result

    def process_directory(self, audio_dir: str,
                          file_pattern: str = "*.wav") -> List[TranscriptionResult]:
        """Process all audio files in a directory."""
        audio_path = Path(audio_dir)
        files = sorted(audio_path.glob(file_pattern))
        logger.info(f"Found {len(files)} audio files in {audio_dir}")

        for i, f in enumerate(files):
            # Extract sentence index from filename (e.g. learner_01_sentence_03)
            match = re.search(r'sentence_(\d+)', f.stem)
            sidx = int(match.group(1)) - 1 if match else None
            self.process_file(str(f), sentence_index=sidx)

        logger.info(f"\nCompleted {len(self.results)} files.")
        return self.results

    def save_results(self, format: str = "both") -> dict:
        """Save results to CSV and/or JSON."""
        saved = {}

        if format in ("csv", "both"):
            csv_path = self.output_dir / "transcriptions.csv"
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                fieldnames = [k for k in asdict(self.results[0]).keys()] if self.results else []
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in self.results:
                    row = asdict(r)
                    row['flags'] = '; '.join(row['flags'])
                    writer.writerow(row)
            saved['csv'] = str(csv_path)
            logger.info(f"Saved CSV: {csv_path}")

        if format in ("json", "both"):
            json_path = self.output_dir / "transcriptions.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump([asdict(r) for r in self.results], f,
                          ensure_ascii=False, indent=2)
            saved['json'] = str(json_path)
            logger.info(f"Saved JSON: {json_path}")

        return saved

    def generate_report(self) -> dict:
        """Generate summary statistics for the transcription run."""
        if not self.results:
            return {}

        confidences = [r.confidence_score for r in self.results]
        durations = [r.duration_seconds for r in self.results]
        proc_times = [r.processing_time_ms for r in self.results]
        flagged = [r for r in self.results if r.flags]

        # Estimate agreement against reference (demo mode uses known refs)
        pp = self.postprocessor
        refs = WhisperTranscriber.EIT_REFERENCE_SENTENCES
        agreements = []
        for r in self.results:
            match = re.search(r'sentence_(\d+)', r.sentence_id)
            if match:
                idx = int(match.group(1)) - 1
                if 0 <= idx < len(refs):
                    score = pp.compute_agreement(r.corrected_transcription, refs[idx])
                    agreements.append(score)

        report = {
            "total_files": len(self.results),
            "avg_confidence": round(np.mean(confidences), 3),
            "min_confidence": round(np.min(confidences), 3),
            "max_confidence": round(np.max(confidences), 3),
            "avg_duration_sec": round(np.mean(durations), 2),
            "avg_processing_ms": round(np.mean(proc_times), 1),
            "flagged_files": len(flagged),
            "flag_rate_pct": round(len(flagged) / len(self.results) * 100, 1),
            "estimated_word_agreement": round(np.mean(agreements), 3) if agreements else "N/A",
            "meets_90pct_target": bool(np.mean(agreements) >= 0.90) if agreements else False
        }

        report_path = self.output_dir / "pipeline_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved: {report_path}")
        return report


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EIT Transcription Pipeline")
    parser.add_argument("--audio_dir", default="audio_samples",
                        help="Directory containing .wav files")
    parser.add_argument("--output_dir", default="outputs",
                        help="Directory to save results")
    parser.add_argument("--model", default="medium",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper model size")
    parser.add_argument("--model_path", default=None,
                        help="Path to fine-tuned model (optional)")
    parser.add_argument("--demo", action="store_true", default=True,
                        help="Run in demo mode (no Whisper needed)")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("  EIT Audio Transcription Pipeline")
    print("="*60 + "\n")

    pipeline = EITPipeline(
        output_dir=args.output_dir,
        use_demo=args.demo,
        whisper_model=args.model,
        model_path=args.model_path
    )

    pipeline.process_directory(args.audio_dir)
    pipeline.save_results(format="both")
    report = pipeline.generate_report()

    print("\n" + "="*60)
    print("  PIPELINE REPORT")
    print("="*60)
    for k, v in report.items():
        print(f"  {k:<35} {v}")
    print("="*60 + "\n")
