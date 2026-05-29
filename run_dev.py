from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from dotenv import load_dotenv
from filelock import FileLock, Timeout


ROOT_DIR = Path(__file__).resolve().parent
API_ENTRYPOINT = ROOT_DIR / "apps" / "api" / "main.py"
WEB_DIR = ROOT_DIR / "apps" / "web"
WEB_PACKAGE = WEB_DIR / "package.json"
ENV_FILE = ROOT_DIR / ".env"
LOCK_FILE = ROOT_DIR / ".run_dev.lock"

load_dotenv(ENV_FILE)


def print_status(message: str) -> None:
    print(f"[runner] {message}")


def require_path(path: Path, label: str) -> None:
    if not path.exists():
        raise RuntimeError(f"Missing {label}: {path}")

def check_required_env() -> None:
    if not os.getenv("DATABASE_URL"):
        raise RuntimeError("DATABASE_URL is not set. Add it to your environment or .env file.")


def check_port(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1.0)
        return sock.connect_ex((host, port)) == 0


def require_free_port(host: str, port: int, label: str) -> None:
    if check_port(host, port):
        raise RuntimeError(
            f"{label} port {port} is already in use on {host}. Stop existing dev server before starting new one."
        )


def check_lm_studio() -> None:
    base_url = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
    health_url = base_url.removesuffix("/v1") + "/v1/models"
    try:
        with urllib.request.urlopen(health_url, timeout=2) as response:
            if response.status != 200:
                print_status(f"Warning: LM Studio returned HTTP {response.status} from {health_url}.")
    except (urllib.error.URLError, TimeoutError):
        print_status(f"Warning: LM Studio did not respond at {health_url}.")


def check_postgres() -> None:
    try:
        import psycopg

        with psycopg.connect(os.environ["DATABASE_URL"], connect_timeout=2) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
    except Exception as exc:
        print_status(f"Warning: PostgreSQL connectivity check failed: {exc}")


def stream_output(process: subprocess.Popen[str], prefix: str) -> threading.Thread:
    def _worker() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            print(f"[{prefix}] {line.rstrip()}")

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return thread


def start_processes() -> list[subprocess.Popen[str]]:
    npm_executable = "npm.cmd" if os.name == "nt" else "npm"
    api_host = os.getenv("BACKEND_HOST", "127.0.0.1")
    api_port = os.getenv("BACKEND_PORT", "8000")

    api_process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "apps.api.main:app",
            "--reload",
            "--host",
            api_host,
            "--port",
            api_port,
        ],
        cwd=ROOT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    web_process = subprocess.Popen(
        [npm_executable, "run", "dev"],
        cwd=WEB_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    stream_output(api_process, "api")
    stream_output(web_process, "web")

    return [api_process, web_process]


def shutdown_processes(processes: list[subprocess.Popen[str]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()

    deadline = time.time() + 5
    for process in processes:
        while process.poll() is None and time.time() < deadline:
            time.sleep(0.1)

    for process in processes:
        if process.poll() is None:
            process.kill()


def main() -> None:
    lock = FileLock(str(LOCK_FILE))
    try:
        lock.acquire(timeout=0.1)
    except Timeout as exc:
        raise RuntimeError(
            "Another run_dev.py instance is already running. Stop it before starting a new dev stack."
        ) from exc

    try:
        print_status("Validating repository layout and dependencies.")
        require_path(API_ENTRYPOINT, "API entrypoint")
        require_path(WEB_PACKAGE, "web package.json")
        check_required_env()
        check_lm_studio()
        check_postgres()
        require_free_port("127.0.0.1", 8000, "API")
        require_free_port("127.0.0.1", 3000, "Web")

        print_status("Starting API and web development servers.")
        processes = start_processes()

        try:
            while True:
                if any(process.poll() is not None for process in processes):
                    raise RuntimeError("One of the dev servers exited unexpectedly.")
                time.sleep(0.5)
        except KeyboardInterrupt:
            print_status("Stopping development servers.")
        finally:
            shutdown_processes(processes)
    finally:
        lock.release()


if __name__ == "__main__":
    main()
