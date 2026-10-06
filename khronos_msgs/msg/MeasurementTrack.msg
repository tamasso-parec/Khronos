# Snapshot of a track, including only its association at this frame's timestamp.
int32 id
bool is_dynamic
bool is_active
float32 confidence
uint64 first_seen_ns
uint64 last_seen_ns
# -1 means there was no corresponding observation in this frame.
int32 semantic_cluster_id
int32 dynamic_cluster_id

