import argparse
import os
import sys
import glob
import time
from typing import List, Dict, Any

import cv2
import pandas as pd
from ultralytics import YOLO


VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm")


def list_videos(input_dir: str) -> List[str]:
    paths = []
    for ext in VIDEO_EXTS:
        paths.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
    return sorted(paths)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def process_video(
    model: YOLO,
    video_path: str,
    out_dir: str,
    save_annotated: bool,
    conf: float,
    device: str,
    use_tracking: bool
) -> Dict[str, Any]:
    """Procesa un video y retorna métricas de conteo."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {
            "video": os.path.basename(video_path),
            "status": "ERROR_OPEN_VIDEO"
        }

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 0.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Preparar writer si se requiere anotado
    writer = None
    annotated_path = None
    if save_annotated:
        ensure_dir(out_dir)
        base = os.path.splitext(os.path.basename(video_path))[0]
        annotated_path = os.path.join(out_dir, f"{base}_annotated.mp4")
        # FourCC y writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(annotated_path, fourcc, fps_in if fps_in > 0 else 25.0, (w, h))

    # Métricas
    frames_processed = 0
    sum_persons = 0               # suma de personas por frame (no es "personas únicas")
    max_concurrent = 0            # pico simultáneo
    unique_ids = set()            # requiere tracking
    t0 = time.time()

    # Procesamiento cuadro a cuadro usando stream
    # Para tracking: model.track(..., tracker="bytetrack.yaml", stream=True)
    generator = None
    if use_tracking:
        generator = model.track(
            source=video_path,
            stream=True,
            conf=conf,
            verbose=False,
            tracker="bytetrack.yaml",
            device=device,
            classes=[0]  # clase 0 = person en COCO
        )
    else:
        generator = model(
            source=video_path,
            stream=True,
            conf=conf,
            verbose=False,
            device=device,
            classes=[0]
        )

    for r in generator:
        # r.orig_img = frame original, r.plot() = frame anotado
        frames_processed += 1

        # Detecciones de personas
        boxes = r.boxes
        count_persons = 0
        if boxes is not None and len(boxes) > 0:
            # Filtrado por clase (0 = person)
            # NOTA: ya limitamos classes=[0], pero validamos por robustez
            for b in boxes:
                if int(b.cls[0]) == 0:
                    count_persons += 1
                    # Si hay tracking, b.id puede existir
                    if use_tracking and hasattr(b, "id") and b.id is not None and len(b.id) > 0:
                        unique_ids.add(int(b.id[0]))

        sum_persons += count_persons
        if count_persons > max_concurrent:
            max_concurrent = count_persons

        # Guardar anotado si aplica
        if save_annotated:
            annotated = r.plot()
            if writer is not None:
                writer.write(annotated)

    # Limpieza
    if writer is not None:
        writer.release()
    cap.release()

    elapsed = time.time() - t0
    duration_s = frames_total / fps_in if fps_in > 0 else None

    return {
        "video": os.path.basename(video_path),
        "status": "OK",
        "frames": frames_processed,
        "fps_in": round(fps_in, 3),
        "width": w,
        "height": h,
        "duration_s_est": round(duration_s, 2) if duration_s else None,
        "sum_persons_over_frames": int(sum_persons),
        "max_concurrent_persons": int(max_concurrent),
        "unique_ids" if use_tracking else "unique_ids (tracking off)": len(unique_ids) if use_tracking else None,
        "annotated_path": annotated_path if save_annotated else None,
        "processing_time_s": round(elapsed, 2)
    }


def main():
    parser = argparse.ArgumentParser(
        description="YOLOv11 - Reconocimiento y Conteo de Personas por lote de videos"
    )
    parser.add_argument("--input_dir", required=True, help="Carpeta con videos")
    parser.add_argument("--output_dir", default="outputs", help="Carpeta de salida (videos anotados y CSV)")
    parser.add_argument("--model", default="yolo11s.pt", help="Modelo YOLOv11 (yolo11n.pt, yolo11s.pt, yolo11m.pt, ...)")
    parser.add_argument("--conf", type=float, default=0.25, help="Umbral de confianza")
    parser.add_argument("--device", default="cpu", help="cpu o cuda:0 (si tienes GPU)")
    parser.add_argument("--save_annotated", action="store_true", help="Guardar videos anotados")
    parser.add_argument("--track", action="store_true", help="Usar tracking (ByteTrack) para contar IDs únicos")
    parser.add_argument("--csv_name", default="people_count_summary.csv", help="Nombre del CSV de resumen")
    args = parser.parse_args()

    vids = list_videos(args.input_dir)
    if not vids:
        print(f"No se encontraron videos en: {args.input_dir}")
        sys.exit(1)

    ensure_dir(args.output_dir)

    print(f"-> Cargando modelo: {args.model}")
    model = YOLO(args.model)

    results = []
    for v in vids:
        print(f"Procesando: {os.path.basename(v)}")
        r = process_video(
            model=model,
            video_path=v,
            out_dir=args.output_dir,
            save_annotated=args.save_annotated,
            conf=args.conf,
            device=args.device,
            use_tracking=args.track
        )
        results.append(r)
        print(f"  Estado: {r.get('status')} | frames: {r.get('frames')} | max_concurrent: {r.get('max_concurrent_persons')} | unique_ids: {r.get('unique_ids') or r.get('unique_ids (tracking off)')}")

    # Guardar CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(args.output_dir, args.csv_name)
    df.to_csv(csv_path, index=False, encoding="utf-8")
    print(f"\nResumen guardado en: {csv_path}")
    print("Listo ✅")


if __name__ == "__main__":
    main()
