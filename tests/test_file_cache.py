"""Tests for the shared file-read cache used in goal pipelines."""

import threading
import time

from coolcode.tools.tracked import FileReadCache


class TestFileReadCache:
    """Tests for FileReadCache — thread-safe shared cache across workers."""

    def test_cache_miss_then_hit(self):
        cache = FileReadCache()
        # First read — miss
        assert cache.get("/foo/bar.py", 0, 2000) is None
        assert cache.misses == 0  # misses counted on put

        # Store
        cache.put("/foo/bar.py", 0, 2000, "file content here")
        assert cache.misses == 1

        # Second read — hit
        result = cache.get("/foo/bar.py", 0, 2000)
        assert result == "file content here"
        assert cache.hits == 1

    def test_different_offsets_are_separate_keys(self):
        cache = FileReadCache()
        cache.put("/foo/bar.py", 0, 100, "first 100 lines")
        cache.put("/foo/bar.py", 100, 100, "next 100 lines")

        assert cache.get("/foo/bar.py", 0, 100) == "first 100 lines"
        assert cache.get("/foo/bar.py", 100, 100) == "next 100 lines"
        assert cache.get("/foo/bar.py", 0, 200) is None  # different limit

    def test_different_files_are_separate(self):
        cache = FileReadCache()
        cache.put("/a.py", 0, 2000, "content A")
        cache.put("/b.py", 0, 2000, "content B")

        assert cache.get("/a.py", 0, 2000) == "content A"
        assert cache.get("/b.py", 0, 2000) == "content B"

    def test_clear_resets_everything(self):
        cache = FileReadCache()
        cache.put("/a.py", 0, 2000, "content")
        cache.get("/a.py", 0, 2000)  # hit
        assert cache.hits == 1
        assert cache.misses == 1

        cache.clear()
        assert cache.hits == 0
        assert cache.misses == 0
        assert cache.get("/a.py", 0, 2000) is None

    def test_thread_safety(self):
        """Multiple threads writing and reading concurrently should not corrupt."""
        cache = FileReadCache()
        errors = []

        def writer(thread_id: int):
            for i in range(50):
                path = f"/file_{thread_id}_{i}.py"
                cache.put(path, 0, 2000, f"content-{thread_id}-{i}")

        def reader(thread_id: int):
            for i in range(50):
                path = f"/file_{thread_id}_{i}.py"
                result = cache.get(path, 0, 2000)
                if result is not None and result != f"content-{thread_id}-{i}":
                    errors.append(f"Corruption: expected content-{thread_id}-{i}, got {result}")

        threads = []
        # 4 writer threads
        for t in range(4):
            threads.append(threading.Thread(target=writer, args=(t,)))
        # Start writers
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Now read — all values should be correct
        reader_threads = []
        for t in range(4):
            reader_threads.append(threading.Thread(target=reader, args=(t,)))
        for t in reader_threads:
            t.start()
        for t in reader_threads:
            t.join()

        assert errors == [], f"Thread safety violations: {errors}"

    def test_hit_miss_counters(self):
        cache = FileReadCache()
        cache.put("/a.py", 0, 100, "a")  # miss=1
        cache.put("/b.py", 0, 100, "b")  # miss=2
        cache.get("/a.py", 0, 100)  # hit=1
        cache.get("/a.py", 0, 100)  # hit=2
        cache.get("/c.py", 0, 100)  # no hit (returns None, not counted)
        cache.get("/b.py", 0, 100)  # hit=3

        assert cache.misses == 2
        assert cache.hits == 3
