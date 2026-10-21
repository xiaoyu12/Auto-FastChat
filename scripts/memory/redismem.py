"""Redis memory provider."""
from typing import List
import redis
from redis.commands.search.field import VectorField, TextField
from redis.commands.search.query import Query
try:
    from redis.commands.search.index_definition import IndexDefinition, IndexType
except ImportError:
    from redis.commands.search.indexDefinition import IndexDefinition, IndexType
import numpy as np

from memory.base import MemoryProviderSingleton, get_ada_embedding


class RedisMemory(MemoryProviderSingleton):
    def __init__(self, cfg):
        """
        Initializes the Redis memory provider.

        Args:
            cfg: The config object.

        Returns: None
        """
        redis_host = cfg.redis_host
        redis_port = cfg.redis_port
        redis_password = cfg.redis_password
        self.dimension = cfg.EMBED_DIM
        self.redis = redis.Redis(
            host=redis_host,
            port=redis_port,
            password=redis_password,
            db=0  # Cannot be changed
        )
        self.cfg = cfg
        if cfg.wipe_redis_on_start:
            self.clear()
        try:
            self.redis.ft(f"{cfg.memory_index}").create_index(
                fields=[
                    TextField("data"),
                    VectorField("embedding", "HNSW", {
                        "TYPE": "FLOAT32",
                        "DIM": self.dimension,
                        "DISTANCE_METRIC": "COSINE",
                    }),
                ],
                definition=IndexDefinition(
                    prefix=[f"{cfg.memory_index}:"],
                    index_type=IndexType.HASH
                )
            )
        except redis.ResponseError as e:
            if str(e).lower() != "index already exists":
                raise

    def add(self, data: str) -> str:
        """
        Adds a data point to the memory.

        Args:
            data: The data to add.

        Returns: Message indicating that the data has been added.
        """
        if 'Command Error:' in data:
            return ""
        vector = self._embedding_bytes(data)
        data_dict = {
            b"data": data,
            "embedding": vector
        }
        # Reserve IDs on the server so separate processes cannot overwrite data.
        vec_num = self.redis.incr(f"{self.cfg.memory_index}-vec_num") - 1
        self.redis.hset(f"{self.cfg.memory_index}:{vec_num}", mapping=data_dict)
        _text = f"Inserting data into memory at index: {vec_num}:\n"\
            f"data: {data}"
        return _text

    def get(self, data: str) -> List[str]:
        """
        Gets the data from the memory that is most relevant to the given data.

        Args:
            data: The data to compare to.

        Returns: The most relevant data.
        """
        return self.get_relevant(data, 1)

    def clear(self) -> str:
        """
        Clears this memory's records, preserving its search index.

        Returns: A message indicating that the memory has been cleared.
        """
        prefix = f"{self.cfg.memory_index}:"
        pattern = "".join(
            "\\" + char if char in "\\*?[]" else char for char in prefix
        ) + "*"
        for key in self.redis.scan_iter(match=pattern):
            self.redis.delete(key)
        # Keep the counter monotonic so active writers cannot reuse an ID.
        return "Obliviated"

    def get_relevant(
        self,
        data: str,
        num_relevant: int = 5
    ) -> List[str]:
        """
        Returns all the data in the memory that is relevant to the given data.
        Args:
            data: The data to compare to.
            num_relevant: The number of relevant data to return.

        Returns: A list of the most relevant data.
        """
        if num_relevant <= 0:
            return []
        query_vector = self._embedding_bytes(data)
        base_query = f"*=>[KNN {num_relevant} @embedding $vector AS vector_score]"
        query = Query(base_query).return_fields(
            "data",
            "vector_score"
        ).sort_by("vector_score").paging(0, num_relevant).dialect(2)

        try:
            results = self.redis.ft(f"{self.cfg.memory_index}").search(
                query, query_params={"vector": query_vector}
            )
        except redis.RedisError as e:
            print("Error calling Redis search: ", e)
            return []
        return [result.data for result in results.docs]

    def _embedding_bytes(self, data: str) -> bytes:
        vector = np.asarray(get_ada_embedding(data), dtype=np.float32)
        if vector.shape != (self.dimension,):
            raise ValueError(
                f"Expected an embedding of dimension {self.dimension}, "
                f"got shape {vector.shape}"
            )
        return vector.tobytes()

    def get_stats(self):
        """
        Returns: The stats of the memory index.
        """
        return self.redis.ft(f"{self.cfg.memory_index}").info()
