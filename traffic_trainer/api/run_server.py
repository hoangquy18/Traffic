#!/usr/bin/env python
"""Script to run the Traffic Prediction API server."""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(description="Run Traffic Prediction API server")
    parser.add_argument(
        "--host", type=str, default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind to (default: 8000)"
    )
    parser.add_argument(
        "--reload", action="store_true", help="Enable auto-reload for development"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model checkpoint (optional, can load via API)",
    )
    parser.add_argument(
        "--config-path", type=str, default=None, help="Path to config.yaml"
    )
    parser.add_argument(
        "--street-info",
        type=str,
        default="traffic_weather_2025_converted.csv",
        help="Path to CSV with street information",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run model on",
    )

    args = parser.parse_args()

    # Pre-load model if path provided
    if args.model_path:
        from traffic_trainer.api.services import model_service

        model_service.load_model(
            checkpoint_path=Path(args.model_path),
            config_path=Path(args.config_path) if args.config_path else None,
            street_info_path=Path(args.street_info) if args.street_info else None,
            device=args.device,
        )

    # Run server
    import uvicorn

    uvicorn.run(
        "traffic_trainer.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
