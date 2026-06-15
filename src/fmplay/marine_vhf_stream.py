from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from fmplay.close_mic import CloseMicError, stream_file
from fmplay.profiles import MarineVhf1993Profile, ProfileError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m fmplay.marine_vhf_stream",
        description=(
            "Stream marine-vhf-1993 audio after an abusive close-mic front end."
        ),
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--squelch-seed", type=int, required=True)
    parser.add_argument("--close-mic-seed", type=int, required=True)
    args = parser.parse_args(argv)

    try:
        profile = MarineVhf1993Profile(
            ffmpeg_command=args.ffmpeg,
            squelch_seed=args.squelch_seed,
            close_mic_seed=args.close_mic_seed,
        )
        command = [
            args.ffmpeg,
            *profile._render_raw_input_args("pipe:0", "pipe:1"),
        ]
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=sys.stdout.buffer,
            stderr=subprocess.PIPE,
        )
        assert process.stdin is not None
        try:
            stream_file(
                args.source,
                process.stdin,
                seed=args.close_mic_seed,
                intensity="abusive",
                ffmpeg_command=args.ffmpeg,
            )
        except (BrokenPipeError, CloseMicError):
            process.kill()
            process.wait()
            raise
        finally:
            try:
                process.stdin.close()
            except BrokenPipeError:
                pass

        stderr = process.stderr.read() if process.stderr is not None else b""
        return_code = process.wait()
        if return_code:
            details = stderr.decode(errors="replace").strip()
            suffix = f": {details.splitlines()[-1]}" if details else ""
            print(f"fmplay-marine-vhf: ffmpeg failed{suffix}", file=sys.stderr)
            return 1
    except BrokenPipeError:
        return 0
    except CloseMicError as exc:
        print(f"fmplay-marine-vhf: {exc}", file=sys.stderr)
        return 1
    except ProfileError as exc:
        print(f"fmplay-marine-vhf: {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
