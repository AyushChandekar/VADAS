"""
VADAS-India — Master Launcher
==============================
Starts the FastAPI backend (which runs the AI inference loop).
The React frontend should be started separately with `npm run dev`.

Usage:
    python run.py                    # default: camera=0, port=8000
    python run.py --camera 1         # use camera index 1
    python run.py --video test.mp4   # use a video file
    python run.py --port 8080        # custom port
"""

import argparse
import os
import sys
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="VADAS-India Launcher")
    parser.add_argument("--camera", default="0",
                        help="Camera source index or URL (default: 0)")
    parser.add_argument("--video", default=None,
                        help="Path to video file (overrides --camera)")
    parser.add_argument("--port", type=int, default=8000,
                        help="Backend API port (default: 8000)")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Backend host (default: 0.0.0.0)")
    args = parser.parse_args()

    # Set camera source via environment variable (read by backend/main.py)
    if args.video:
        if not os.path.exists(args.video):
            print(f"Error: Video file not found: {args.video}")
            sys.exit(1)
        os.environ["CAMERA_SOURCE"] = args.video
    else:
        os.environ["CAMERA_SOURCE"] = args.camera

    print("=" * 60)
    print("  VADAS-India — Vehicle Autonomous Driving Assistance")
    print("=" * 60)
    print(f"  Backend:  http://{args.host}:{args.port}")
    print(f"  Camera:   {args.video or args.camera}")
    print(f"  Dashboard: Start separately with:")
    print(f"    cd frontend && npm install && npm run dev")
    print(f"    Then open http://localhost:3000")
    print("=" * 60)

    uvicorn.run(
        "backend.main:app",
        host=args.host,
        port=args.port,
        reload=False,
        workers=1,
    )


if __name__ == "__main__":
    main()
