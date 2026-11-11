import functools
import time


def retry(times=3, delay=1.5):
    """Retry transient failures with backoff; client errors (400/401/403/404) are raised at once."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            for attempt in range(times):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    if attempt == times - 1 or getattr(e, "status_code", None) in (400, 401, 403, 404):
                        raise
                    time.sleep(delay * 2 ** attempt)
        return wrapper
    return deco
