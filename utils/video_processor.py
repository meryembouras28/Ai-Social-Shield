import json
import shutil
import subprocess
import uuid
from pathlib import Path

import requests
import yt_dlp


TEMP_DIR = Path("temp/video_processing")
TEMP_DIR.mkdir(parents=True, exist_ok=True)


class VideoProcessor:

    def __init__(self):
        self.ffmpeg_path = shutil.which("ffmpeg")
        self.ffprobe_path = shutil.which("ffprobe")

        if self.ffmpeg_path is None:
            raise RuntimeError(
                "FFmpeg est introuvable. Installez FFmpeg et ajoutez-le au PATH."
            )

        if self.ffprobe_path is None:
            raise RuntimeError(
                "ffprobe est introuvable. Il est normalement fourni avec FFmpeg."
            )

    def create_work_dir(self):
        work_dir = TEMP_DIR / str(uuid.uuid4())
        work_dir.mkdir(parents=True, exist_ok=True)
        return work_dir

    def download_video(self, video_url):
        work_dir = self.create_work_dir()

        output_template = str(work_dir / "video.%(ext)s")

        ydl_opts = {
            "format": "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
            "outtmpl": output_template,
            "merge_output_format": "mp4",
            "quiet": False,
            "noplaylist": True,
            "restrictfilenames": True,
            "retries": 10,
            "fragment_retries": 10,
            "file_access_retries": 5,
            "extractor_retries": 5,
            "socket_timeout": 60,
            "http_chunk_size": 10485760,
            "concurrent_fragment_downloads": 1,
            "continuedl": True,
            "nopart": False,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(
                video_url,
                download=True,
            )

            downloaded_path = Path(
                ydl.prepare_filename(info)
            )

        mp4_path = work_dir / "video.mp4"

        if mp4_path.exists():
            downloaded_path = mp4_path

        if not downloaded_path.exists():
            files = list(work_dir.glob("video.*"))

            video_candidates = [
                path
                for path in files
                if path.suffix.lower()
                not in {
                    ".vtt",
                    ".srt",
                    ".ass",
                    ".json",
                    ".part",
                }
            ]

            if not video_candidates:
                raise RuntimeError(
                    "La vidéo n'a pas pu être téléchargée."
                )

            downloaded_path = video_candidates[0]

        subtitle_opts = {
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["all"],
            "subtitlesformat": "vtt/best",
            "quiet": True,
            "no_warnings": True,
            "ignoreerrors": True,
            "noplaylist": True,
            "restrictfilenames": True,
            "socket_timeout": 30,
            "retries": 2,
        }

        try:
            with yt_dlp.YoutubeDL(subtitle_opts) as ydl:
                ydl.download([video_url])
        except Exception:
            pass

        subtitle_path = self.find_subtitle(work_dir)

        return {
            "video_path": downloaded_path,
            "work_dir": work_dir,
            "title": info.get("title"),
            "duration": info.get("duration"),
            "subtitle_path": subtitle_path,
        }

    def find_subtitle(self, work_dir):
        work_dir = Path(work_dir)

        subtitle_extensions = [".vtt", ".srt", ".ass"]

        candidates = []

        for extension in subtitle_extensions:
            candidates.extend(
                work_dir.glob(f"*{extension}")
            )

        if not candidates:
            return None

        return candidates[0]

    def download_external_file(self, url, destination_path):
        destination_path = Path(destination_path)

        response = requests.get(
            url,
            stream=True,
            timeout=30,
            headers={
                "User-Agent": "Mozilla/5.0 AI-Social-Shield"
            },
        )

        response.raise_for_status()

        content_length = response.headers.get(
            "content-length"
        )

        if content_length:
            content_length = int(content_length)

            if content_length > 20 * 1024 * 1024:
                raise RuntimeError(
                    "Le fichier externe dépasse la taille maximale autorisée de 20 MB."
                )

        downloaded_size = 0

        with destination_path.open("wb") as output_file:
            for chunk in response.iter_content(
                chunk_size=1024 * 1024
            ):
                if not chunk:
                    continue

                downloaded_size += len(chunk)

                if downloaded_size > 20 * 1024 * 1024:
                    raise RuntimeError(
                        "Le fichier externe dépasse la taille maximale autorisée de 20 MB."
                    )

                output_file.write(chunk)

        if (
            not destination_path.exists()
            or destination_path.stat().st_size == 0
        ):
            raise RuntimeError(
                "Le fichier externe est vide."
            )

        return destination_path

    def get_duration(self, video_path):
        video_path = Path(video_path)

        command = [
            self.ffprobe_path,
            "-v",
            "quiet",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(video_path),
        ]

        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
        )

        metadata = json.loads(result.stdout)

        duration = metadata["format"].get(
            "duration",
            0,
        )

        return float(duration)

    def extract_thumbnail(
        self,
        video_path,
        timestamp=1.0,
    ):
        video_path = Path(video_path)

        thumbnail_path = (
            video_path.parent / "thumbnail.jpg"
        )

        command = [
            self.ffmpeg_path,
            "-y",
            "-ss",
            str(timestamp),
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            "-q:v",
            "2",
            str(thumbnail_path),
        ]

        subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )

        if not thumbnail_path.exists():
            raise RuntimeError(
                "Impossible d'extraire la miniature."
            )

        return thumbnail_path

    def extract_key_frames(
        self,
        video_path,
        frame_count=5,
    ):
        video_path = Path(video_path)

        duration = self.get_duration(video_path)

        if duration <= 0:
            raise RuntimeError(
                "Durée vidéo invalide."
            )

        frame_count = max(
            1,
            min(frame_count, 5),
        )

        frames_dir = (
            video_path.parent / "frames"
        )

        frames_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        timestamps = [
            duration * (i + 1) / (frame_count + 1)
            for i in range(frame_count)
        ]

        frame_paths = []

        for index, timestamp in enumerate(
            timestamps,
            start=1,
        ):
            frame_path = (
                frames_dir / f"frame_{index}.jpg"
            )

            command = [
                self.ffmpeg_path,
                "-y",
                "-ss",
                str(timestamp),
                "-i",
                str(video_path),
                "-frames:v",
                "1",
                "-q:v",
                "2",
                str(frame_path),
            ]

            subprocess.run(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )

            if frame_path.exists():
                frame_paths.append(frame_path)

        return frame_paths

    def extract_audio(self, video_path):
        video_path = Path(video_path)

        audio_path = (
            video_path.parent / "audio.wav"
        )

        command = [
            self.ffmpeg_path,
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_s16le",
            str(audio_path),
        ]

        result = subprocess.run(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )

        if (
            result.returncode != 0
            or not audio_path.exists()
        ):
            return None

        return audio_path

    def cleanup(self, work_dir):
        if work_dir is None:
            return

        work_dir = Path(work_dir)

        if work_dir.exists():
            shutil.rmtree(
                work_dir,
                ignore_errors=True,
            )