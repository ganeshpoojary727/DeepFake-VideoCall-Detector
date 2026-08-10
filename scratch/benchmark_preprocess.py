"""Benchmark parallel face cropping and pre-processing speed."""

import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from app.video.preprocessing.video_decoder import VideoDecoder
from app.video.preprocessing.frame_sampler import FrameSampler
from app.video.face_detection.face_detector import FaceDetector

def process_single_video(vpath_str: str) -> bool:
    try:
        decoder = VideoDecoder()
        sampler = FrameSampler(num_frames=16, strategy="uniform")
        detector = FaceDetector()

        frames = decoder.decode(vpath_str)
        sampled = sampler.sample(frames)

        crops = []
        for f in sampled:
            bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
            box = detector.detect_largest(bgr)
            if box is not None:
                cropped = detector.crop_face(bgr, box, target_size=(224, 224))
            else:
                h, w = f.shape[:2]
                s = min(h, w)
                crop = cv2.resize(f[(h-s)//2:(h+s)//2, (w-s)//2:(w+s)//2], (224, 224))
                cropped = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
            crops.append(cropped)
        return True
    except Exception:
        return False

def main():
    FFPP_DIR = PROJECT_ROOT / "datasets" / "video" / "faceforensics" / "original"
    videos = [str(p) for p in sorted(list(FFPP_DIR.glob("*.mp4")))[:100]]

    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_single_video, vp) for vp in videos]
        results = [f.result() for f in as_completed(futures)]
    t1 = time.perf_counter()

    print(f"Processed 100 videos with 8 workers in {t1 - t0:.2f}s ({(t1 - t0)/100:.3f}s per video)")
    print(f"Extrapolated time for 6,340 total videos: {(t1 - t0)/100 * 6340 / 60:.2f} minutes")

if __name__ == "__main__":
    main()
