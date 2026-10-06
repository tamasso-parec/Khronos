# One processed active-window frame, exported after tracking and integration.
# Header frame is the world coordinate frame used by the input poses.
std_msgs/Header header
# Starts at zero for each publisher instance, increments even without subscribers.
uint64 sequence
geometry_msgs/Pose world_t_sensor
# Processed geometry: planar depth in meters and exact world-frame XYZ vertices.
sensor_msgs/Image depth
sensor_msgs/Image vertices
# Remapped semantic categories and frame-local detection IDs, all 32SC1.
# Zero in dynamic/object images means no accepted detection, not known static.
sensor_msgs/Image semantic_labels
sensor_msgs/Image dynamic_labels
sensor_msgs/Image object_labels
khronos_msgs/MeasurementTrack[] tracks
