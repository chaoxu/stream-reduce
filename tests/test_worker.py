"""Exercise delayed retries through the real worker using an isolated Redis."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
from redis import Redis
from redis.exceptions import ConnectionError
from rq import Queue, Retry, get_current_job
from rq.job import JobStatus


def fail_once() -> int:
    job = get_current_job()
    job.connection.rpush("test:worker-pids", os.getpid())
    if job.connection.incr("test:attempts") == 1:
        raise RuntimeError("transient test failure")
    return 2


def _stop(process: subprocess.Popen) -> None:
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


@pytest.fixture
def isolated_redis():
    executable = shutil.which("redis-server")
    if executable is None:
        pytest.skip("redis-server is required for the worker integration test")
    # A short path also works with macOS's Unix-socket path limit.
    with tempfile.TemporaryDirectory(prefix="sr-rq-", dir="/tmp") as directory:
        socket = Path(directory) / "redis.sock"
        process = subprocess.Popen(
            [
                executable,
                "--port",
                "0",
                "--unixsocket",
                str(socket),
                "--unixsocketperm",
                "700",
                "--save",
                "",
                "--appendonly",
                "no",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        connection = Redis(unix_socket_path=str(socket), protocol=2)
        try:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                assert process.poll() is None, "isolated Redis exited before startup"
                try:
                    connection.ping()
                    break
                except ConnectionError:
                    time.sleep(0.05)
            else:
                pytest.fail("isolated Redis did not start")
            yield connection, f"unix://{socket}?protocol=2"
        finally:
            connection.close()
            _stop(process)


def test_worker_executes_delayed_retry(isolated_redis, tmp_path):
    connection, redis_url = isolated_redis
    queue = Queue("stream-reduce", connection=connection)
    job = queue.enqueue(fail_once, retry=Retry(max=1, interval=2))
    root = Path(__file__).resolve().parents[1]
    env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(root), str(root / "tests")]),
        "REDIS_URL": redis_url,
        "DATA_DIR": str(tmp_path / "data"),
        "DATABASE_URL": "",
        "MEDIA_DIR": str(tmp_path / "media"),
        "ENABLE_EMBEDDINGS": "false",
    }
    with (tmp_path / "worker.log").open("w+") as log:
        worker = subprocess.Popen(
            [sys.executable, "-m", "worker.run"],
            cwd=tmp_path,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            saw_scheduled = False
            deadline = time.monotonic() + 12
            while time.monotonic() < deadline and worker.poll() is None:
                status = job.get_status(refresh=True)
                saw_scheduled |= status == JobStatus.SCHEDULED
                if status == JobStatus.FINISHED:
                    break
                time.sleep(0.05)
            log.seek(0)
            assert job.get_status(refresh=True) == JobStatus.FINISHED, log.read()
            assert saw_scheduled, "the failure must pass through delayed scheduling"
            assert job.return_value() == 2
            assert connection.get("test:attempts") == b"2"
            # SimpleWorker must still execute both attempts serially in its own process.
            assert connection.lrange("test:worker-pids", 0, -1) == [
                str(worker.pid).encode(),
                str(worker.pid).encode(),
            ]
            assert queue.scheduled_job_registry.count == 0
        finally:
            _stop(worker)
