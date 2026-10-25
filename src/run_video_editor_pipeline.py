"""Generate subtitles for a video.

Example:
    python src/run_video_editor_pipeline.py data/videos/sample_video.mp4
    python src/run_video_editor_pipeline.py data/videos/sample_video.mp4 --language en
"""

import argparse

from mindbase_layer.tools import generate_subtitles


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate subtitles for a video.")
    parser.add_argument("input_file", help="Path to the input video file")
    parser.add_argument(
        "--language",
        default="ru",
        help="Subtitle language (default: ru)",
    )
    args = parser.parse_args()

    print("Starting the video editor agent...")
    generate_subtitles(video_path=args.input_file, language=args.language)

if __name__ == "__main__":
    main()