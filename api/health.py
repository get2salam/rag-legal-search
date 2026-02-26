"""
Health check and monitoring endpoints.

Provides Kubernetes-compatible liveness and readiness probes,
plus a comprehensive health report for debugging.

Endpoints:
    GET /health          → Full health report
    GET /health/live     → Liveness probe (is the process alive?)
    GET /health/ready    → Readiness probe (can it serve traffic?)
    GET /metrics         → Prometheus-format metrics
    GET /metrics/json    → JSON metrics snapshot
"""

import os
import time
import platform
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from utils.metrics import get_collector


class HealthStatus(str, Enum):
    """Health status values."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class DependencyCheck(BaseModel):
    """Health status of a single dependency."""

    name: str
    status: HealthStatus
    latency_ms: Optional[float] = None
    message: Optional[str] = None
    details: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    """Full health check response."""

    status: HealthStatus
    version: str
    uptime_seconds: float
    timestamp: str
    hostname: str
    python_version: str
    dependencies: List[DependencyCheck]


# Track startup time
_start_time = time.time()
_version = os.environ.get("APP_VERSION", "1.0.0")


def _check_vector_store() -> DependencyCheck:
    """Check vector store connectivity."""
    start = time.perf_counter()
    try:
        chroma_host = os.environ.get("CHROMA_HOST", "localhost")
        chroma_port = os.environ.get("CHROMA_PORT", "8000")

        import urllib.request

        url = f"http://{chroma_host}:{chroma_port}/api/v1/heartbeat"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as _:
            latency = (time.perf_counter() - start) * 1000
            return DependencyCheck(
                name="chromadb",
                status=HealthStatus.HEALTHY,
                latency_ms=round(latency, 2),
                message="Connected",
            )
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        return DependencyCheck(
            name="chromadb",
            status=HealthStatus.UNHEALTHY,
            latency_ms=round(latency, 2),
            message=f"Connection failed: {type(e).__name__}",
        )


def _check_embedding_service() -> DependencyCheck:
    """Check embedding model availability."""
    start = time.perf_counter()
    try:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            return DependencyCheck(
                name="embedding_service",
                status=HealthStatus.DEGRADED,
                message="OPENAI_API_KEY not configured; demo mode only",
            )

        latency = (time.perf_counter() - start) * 1000
        return DependencyCheck(
            name="embedding_service",
            status=HealthStatus.HEALTHY,
            latency_ms=round(latency, 2),
            message="API key configured",
        )
    except Exception as e:
        latency = (time.perf_counter() - start) * 1000
        return DependencyCheck(
            name="embedding_service",
            status=HealthStatus.UNHEALTHY,
            latency_ms=round(latency, 2),
            message=str(e),
        )


def _check_disk_space() -> DependencyCheck:
    """Check available disk space."""
    try:
        import shutil

        usage = shutil.disk_usage("/")
        free_gb = usage.free / (1024**3)
        total_gb = usage.total / (1024**3)
        pct_free = (usage.free / usage.total) * 100

        status = HealthStatus.HEALTHY
        if pct_free < 5:
            status = HealthStatus.UNHEALTHY
        elif pct_free < 15:
            status = HealthStatus.DEGRADED

        return DependencyCheck(
            name="disk_space",
            status=status,
            message=f"{free_gb:.1f}GB free of {total_gb:.1f}GB ({pct_free:.1f}%)",
            details={"free_gb": round(free_gb, 2), "total_gb": round(total_gb, 2)},
        )
    except Exception as e:
        return DependencyCheck(
            name="disk_space",
            status=HealthStatus.DEGRADED,
            message=str(e),
        )


def create_health_app() -> FastAPI:
    """
    Create FastAPI application with health and metrics endpoints.

    Returns:
        Configured FastAPI app
    """
    app = FastAPI(
        title="RAG Legal Search — Monitoring",
        description="Health checks and metrics for the RAG Legal Search service",
        version=_version,
        docs_url="/docs",
        redoc_url=None,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """
        Comprehensive health check.

        Returns full status including all dependency checks.
        Useful for dashboards and debugging.
        """
        checks = [
            _check_vector_store(),
            _check_embedding_service(),
            _check_disk_space(),
        ]

        # Aggregate status
        statuses = {c.status for c in checks}
        if HealthStatus.UNHEALTHY in statuses:
            overall = HealthStatus.UNHEALTHY
        elif HealthStatus.DEGRADED in statuses:
            overall = HealthStatus.DEGRADED
        else:
            overall = HealthStatus.HEALTHY

        return HealthResponse(
            status=overall,
            version=_version,
            uptime_seconds=round(time.time() - _start_time, 2),
            timestamp=datetime.now(timezone.utc).isoformat(),
            hostname=platform.node(),
            python_version=platform.python_version(),
            dependencies=checks,
        )

    @app.get("/health/live")
    async def liveness():
        """
        Kubernetes liveness probe.

        Returns 200 if the process is alive. Only fails if the
        application is in an unrecoverable state.
        """
        return {"status": "alive"}

    @app.get("/health/ready")
    async def readiness():
        """
        Kubernetes readiness probe.

        Returns 200 only if the service can handle requests.
        Checks critical dependencies.
        """
        vector_check = _check_vector_store()

        if vector_check.status == HealthStatus.UNHEALTHY:
            return Response(
                content='{"status": "not ready", "reason": "vector store unavailable"}',
                media_type="application/json",
                status_code=503,
            )

        return {"status": "ready"}

    @app.get("/metrics")
    async def prometheus_metrics():
        """
        Export metrics in Prometheus text exposition format.

        Compatible with Prometheus scrape targets and
        Grafana dashboards.
        """
        collector = get_collector()
        return Response(
            content=collector.to_prometheus(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    @app.get("/metrics/json")
    async def json_metrics():
        """
        Export metrics as JSON.

        Useful for custom dashboards and debugging.
        """
        collector = get_collector()
        return collector.snapshot()

    return app


# Module-level app instance for uvicorn
monitoring_app = create_health_app()
