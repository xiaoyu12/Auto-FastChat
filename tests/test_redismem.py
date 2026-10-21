"""Redis regression tests; only the search module and embeddings are mocked."""

import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import fakeredis
import numpy as np
import pytest
import redis


@pytest.fixture
def redismem(monkeypatch):
    # Avoid loading model weights and the other memory backends at import time.
    base = ModuleType("memory.base")
    base.MemoryProviderSingleton = object
    base.get_ada_embedding = Mock(return_value=[1.0, 0.0, 0.0])
    monkeypatch.setitem(sys.modules, "memory.base", base)
    path = Path(__file__).resolve().parents[1] / "scripts/memory/redismem.py"
    spec = importlib.util.spec_from_file_location("redismem_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def backend(redismem, monkeypatch):
    client = fakeredis.FakeRedis()
    search = Mock()
    monkeypatch.setattr(client, "ft", Mock(return_value=search))
    monkeypatch.setattr(redismem.redis, "Redis", Mock(return_value=client))
    cfg = SimpleNamespace(
        redis_host="localhost", redis_port=6379, redis_password="",
        memory_index="memory", wipe_redis_on_start=False, EMBED_DIM=3,
    )
    return SimpleNamespace(
        client=client, search=search, cfg=cfg,
        new=lambda: redismem.RedisMemory(cfg),
    )


def test_import_supports_current_redis_py(redismem):
    assert redismem.RedisMemory is not None


@pytest.mark.parametrize("on_start", [False, True])
def test_clear_preserves_other_keys_and_search_index(backend, on_start):
    backend.client.hset("memory:0", mapping={"data": "old"})
    backend.client.set("memory-vec_num", 1)
    backend.client.hset("other:0", mapping={"data": "keep"})
    backend.client.set("application-setting", "keep")
    backend.cfg.wipe_redis_on_start = on_start
    memory = backend.new()
    backend.search.reset_mock()

    if not on_start:
        assert memory.clear() == "Obliviated"

    assert not backend.client.exists("memory:0")
    assert backend.client.hget("other:0", "data") == b"keep"
    assert backend.client.get("application-setting") == b"keep"
    backend.search.dropindex.assert_not_called()
    memory.add("new")
    assert backend.client.hget("memory:1", "data") == b"new"


def test_clear_treats_index_name_as_literal(backend):
    backend.cfg.memory_index = r"memory[*?]\name"
    key = backend.cfg.memory_index + ":0"
    backend.client.hset(key, mapping={"data": "delete"})
    backend.client.hset("memoryx:0", mapping={"data": "keep"})

    backend.new().clear()

    assert not backend.client.exists(key)
    assert backend.client.hget("memoryx:0", "data") == b"keep"


def test_independent_writers_do_not_overwrite_memories(backend):
    first = backend.new()
    backend.search.create_index.side_effect = redis.ResponseError(
        "Index already exists"
    )
    second = backend.new()

    first.add("first")
    second.add("second")
    first.add("third")

    assert [backend.client.hget(f"memory:{i}", "data") for i in range(3)] == [
        b"first", b"second", b"third",
    ]
    assert backend.client.get("memory-vec_num") == b"3"


def test_resumes_existing_record_counter(backend):
    backend.client.hset("memory:4", mapping={"data": "existing"})
    backend.client.set("memory-vec_num", 5)

    message = backend.new().add("new")

    assert "index: 5:" in message
    assert backend.client.hget("memory:4", "data") == b"existing"
    assert backend.client.hget("memory:5", "data") == b"new"


def test_schema_uses_configured_embedding_dimension(backend):
    backend.new()

    fields = backend.search.create_index.call_args.kwargs["fields"]
    args = fields[1].redis_args()
    assert args[args.index("DIM") + 1] == 3


@pytest.mark.parametrize("error", [
    redis.ResponseError("unknown command FT.CREATE"),
    redis.ConnectionError("unreachable"),
    RuntimeError("unexpected failure"),
])
def test_unexpected_index_creation_errors_are_not_suppressed(backend, error):
    backend.search.create_index.side_effect = error

    with pytest.raises(type(error), match=str(error)):
        backend.new()


def test_existing_index_is_allowed(backend):
    backend.search.create_index.side_effect = redis.ResponseError(
        "Index already exists"
    )

    backend.new().add("new")

    assert backend.client.hget("memory:0", "data") == b"new"


def test_search_requests_all_results_and_serializes_float32(backend):
    expected = [f"result {i}" for i in range(15)]
    backend.search.search.return_value = SimpleNamespace(
        docs=[SimpleNamespace(data=value) for value in expected]
    )

    assert backend.new().get_relevant("query", 15) == expected

    args, kwargs = backend.search.search.call_args
    query_args = args[0].get_args()
    limit = query_args.index("LIMIT")
    assert list(map(int, query_args[limit + 1:limit + 3])) == [0, 15]
    assert kwargs["query_params"]["vector"] == np.array(
        [1.0, 0.0, 0.0], dtype=np.float32
    ).tobytes()


@pytest.mark.parametrize("count", [0, -1])
def test_nonpositive_result_count_skips_embedding_and_search(
    backend, redismem, count
):
    assert backend.new().get_relevant("query", count) == []

    redismem.get_ada_embedding.assert_not_called()
    backend.search.search.assert_not_called()


def test_redis_search_errors_return_an_empty_list(backend):
    backend.search.search.side_effect = redis.ConnectionError("unreachable")

    assert backend.new().get("query") == []


@pytest.mark.parametrize("method", ["add", "get"])
@pytest.mark.parametrize("embedding", [[1.0, 0.0], [[1.0, 0.0, 0.0]]])
def test_wrong_embedding_shape_is_rejected(
    backend, redismem, method, embedding
):
    redismem.get_ada_embedding.return_value = embedding

    with pytest.raises(ValueError, match="dimension"):
        getattr(backend.new(), method)("text")

    assert backend.client.dbsize() == 0
    backend.search.search.assert_not_called()


def test_command_errors_are_not_stored(backend, redismem):
    assert backend.new().add("Command Error: failed") == ""

    assert backend.client.dbsize() == 0
    redismem.get_ada_embedding.assert_not_called()
