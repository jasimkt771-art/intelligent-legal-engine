from cache import clear_cache, connect_to_semantic_cache, connect_to_redis, connect_to_cache

redis, semantic = connect_to_cache()
print(semantic)