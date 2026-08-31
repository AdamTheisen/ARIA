from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
import json
import os
import pickle
import shutil
import tempfile
import time


DEFAULT_CACHE_DIR = Path.home() / ".cache" / "aria" / "processed"


class PersistentCache:
    """
    Lightweight persistent cache for expensive GPGL derived products.

    Objects are stored as local pickle payloads created by this package. Cache
    keys are hashed into filenames; a JSON sidecar preserves readable metadata.
    Writes are atomic so interrupted Streamlit sessions do not leave partial
    cache entries.
    """

    def __init__(self, root=None):
        self.root = Path(root or DEFAULT_CACHE_DIR)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _digest(key: str) -> str:
        return sha256(key.encode("utf-8")).hexdigest()[:24]

    def _paths(self, namespace: str, key: str):
        ns = self.root / namespace
        ns.mkdir(parents=True, exist_ok=True)
        stem = self._digest(key)
        return ns / f"{stem}.pkl", ns / f"{stem}.json"

    def get(self, namespace: str, key: str, *, max_age_seconds=None):
        payload, meta = self._paths(namespace, key)
        if not payload.exists():
            return None

        if max_age_seconds is not None:
            age = time.time() - payload.stat().st_mtime
            if age > max_age_seconds:
                return None

        try:
            with payload.open("rb") as handle:
                return pickle.load(handle)
        except Exception:
            # Corrupt or incompatible local cache entries are disposable.
            try:
                payload.unlink(missing_ok=True)
                meta.unlink(missing_ok=True)
            except Exception:
                pass
            return None

    def set(self, namespace: str, key: str, value, *, metadata=None):
        payload, meta = self._paths(namespace, key)
        tmp = payload.with_suffix(".tmp")

        with tmp.open("wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, payload)

        info = {
            "key": key,
            "namespace": namespace,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
        if metadata:
            info.update(metadata)

        meta_tmp = meta.with_suffix(".tmp")
        meta_tmp.write_text(json.dumps(info, indent=2, default=str))
        os.replace(meta_tmp, meta)
        self.prune()
        return value

    def get_or_build(
        self,
        namespace: str,
        key: str,
        builder,
        *,
        max_age_seconds=None,
        metadata=None,
        force=False,
    ):
        if not force:
            cached = self.get(
                namespace,
                key,
                max_age_seconds=max_age_seconds,
            )
            if cached is not None:
                return cached, True

        value = builder()
        self.set(namespace, key, value, metadata=metadata)
        return value, False

    def prune(self, *, max_age_seconds=3*24*3600, max_bytes=2*1024**3):
        """
        Bound cache growth. Old files are removed first, then the oldest
        remaining entries are removed until the requested size cap is met.
        """
        if not self.root.exists():
            return

        now=time.time()
        payloads=sorted(
            self.root.rglob("*.pkl"),
            key=lambda p:p.stat().st_mtime,
        )

        for payload in list(payloads):
            try:
                if now-payload.stat().st_mtime > max_age_seconds:
                    payload.unlink(missing_ok=True)
                    payload.with_suffix(".json").unlink(missing_ok=True)
            except Exception:
                pass

        payloads=sorted(
            [p for p in self.root.rglob("*.pkl") if p.exists()],
            key=lambda p:p.stat().st_mtime,
        )
        total=sum(p.stat().st_size for p in payloads)
        for payload in payloads:
            if total <= max_bytes:
                break
            try:
                size=payload.stat().st_size
                payload.unlink(missing_ok=True)
                payload.with_suffix(".json").unlink(missing_ok=True)
                total-=size
            except Exception:
                pass

    def clear(self, namespace=None):
        target = self.root if namespace is None else self.root / namespace
        if target.exists():
            shutil.rmtree(target)
        if namespace is None:
            self.root.mkdir(parents=True, exist_ok=True)

    def size_bytes(self):
        if not self.root.exists():
            return 0
        return sum(
            p.stat().st_size
            for p in self.root.rglob("*")
            if p.is_file()
        )


def time_bucket(minutes: int, when=None) -> str:
    """Return a UTC cache bucket label such as 20260828T1820Z."""
    now = datetime.now(timezone.utc) if when is None else when
    if hasattr(now, "to_pydatetime"):
        now = now.to_pydatetime()
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    else:
        now = now.astimezone(timezone.utc)
    minute = (now.minute // int(minutes)) * int(minutes)
    bucket = now.replace(minute=minute, second=0, microsecond=0)
    return bucket.strftime("%Y%m%dT%H%MZ")
