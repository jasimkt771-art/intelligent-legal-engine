from cache import connect_to_cache, clear_cache

redis, semantic = connect_to_cache()
clear_cache(redis, semantic)