# URLab Image Transport Relay

Small ROS2 C++ sidecar that relays a URLab raw `sensor_msgs/Image` topic through
the standard `image_transport` publisher stack. Use it when you want official
`image_transport` plugins such as `compressed`, `compressedDepth`, `theora`,
`zstd`, or `turbojpeg` instead of the Python bridge's direct
`sensor_msgs/CompressedImage` publisher.

This package now provides two executables:

- `urlab_camera_bridge`: subscribes directly to URLab camera ZMQ publishers and
  publishes through `image_transport`.
- `image_transport_relay`: compatibility/debug relay from an existing ROS2 raw
  `sensor_msgs/Image` topic into `image_transport`.

Prefer `urlab_camera_bridge` for camera performance work. It removes the extra
Python raw-image publisher from the camera path while leaving Python in charge
of Go2 policy, web control, ROS2 `cmd_vel`, state, and odometry.

## Build

```bash
cd /sata/axin/URLab_Bridge
source /opt/ros/jazzy/setup.bash

sudo apt install \
  libzmq3-dev \
  libmsgpack-cxx-dev \
  ros-jazzy-image-transport-plugins \
  ros-jazzy-compressed-image-transport \
  ros-jazzy-turbojpeg-compressed-image-transport

colcon build \
  --base-paths ros2/urlab_image_transport_relay \
  --symlink-install

source install/setup.bash
```

## Direct URLab Camera Bridge

Start `URLab_Bridge` without Python ROS2 camera publishing:

```bash
python scripts/run_go2_moe_multi_web.py \
  --web-target go2_go2_rl_gym_C_1:8099 \
  --web-camera go2_go2_rl_gym_C_1:front_rgb
```

Then publish the selected URLab camera through `image_transport`:

```bash
ros2 run urlab_image_transport_relay urlab_camera_bridge --ros-args \
  -p urlab_address:=tcp://127.0.0.1 \
  -p step_port:=5559 \
  -p cameras:="['go2_go2_rl_gym_C_1:front_rgb']" \
  -p queue_size:=1 \
  -p output_suffix:=image_transport \
  -p log_interval_s:=1.0
```

The bridge reads URLab camera metadata from the sessionless
`describe_runtime` RPC and subscribes to the advertised per-camera ZMQ
endpoint/topic. This deliberately leaves the Python policy server's active
`hello` session untouched. Cameras must have `Broadcast to ZMQ` enabled in
URLab.

`auto_enable_cameras` remains a deprecated compatibility parameter, but setting
it to `true` now exits before any URLab RPC. Camera capture is owned by URLab
UI/Blueprint or the existing controller session; the direct bridge is a
read-only camera reader.

For a `real` camera, current URLab runtime metadata advertises
`payload_encoding=bgra8_srgb`. The bridge performs BGRA-to-ROS-RGB channel
reordering only. It also accepts the legacy `bgra8_linear` contract, for which
it applies one linear-to-sRGB LUT before that reorder. Do not apply a second
gamma transfer in a downstream consumer of the resulting `rgb8` topic.

With image transport plugins installed, common topics are:

```text
/go2_go2_rl_gym_C_1/camera/front_rgb/image_transport
/go2_go2_rl_gym_C_1/camera/front_rgb/image_transport/compressed
/go2_go2_rl_gym_C_1/camera/front_rgb/image_transport/zstd
/go2_go2_rl_gym_C_1/camera/front_rgb/camera_info
```

`camera_info` is a calibrated centered pinhole model built from URLab's
advertised resolution and MuJoCo vertical `fovy`. The bridge assumes square
pixels and zero lens distortion, publishes `distortion_model=plumb_bob`, and
fills `D`, `K`, `R`, and `P`. Invalid resolution or FOV metadata stops bridge
startup instead of publishing false calibration. For 640x480 at 90 degrees
vertical FOV, `fx=fy=240`, `cx=319.5`, and `cy=239.5`.

## Compatibility Relay

The relay must use distinct input and output topics. If it subscribes to and
publishes the same base topic, the raw transport plugin can feed the relay's own
output back into its input.

Start `URLab_Bridge` with raw camera publishing enabled:

```bash
python scripts/run_go2_moe_multi_web.py \
  --web-target go2_go2_rl_gym_C_1:8099 \
  --web-camera go2_go2_rl_gym_C_1:front_rgb \
  --ros2-publish-cameras \
  --ros2-camera-fps 60
```

Then relay that raw topic through `image_transport`:

```bash
ros2 run urlab_image_transport_relay image_transport_relay --ros-args \
  -p input_topic:=/go2_go2_rl_gym_C_1/camera/front_rgb/image_raw \
  -p output_topic:=/go2_go2_rl_gym_C_1/camera/front_rgb/image_transport \
  -p queue_size:=1
```

With `compressed_image_transport` installed, the compressed topic is:

```text
/go2_go2_rl_gym_C_1/camera/front_rgb/image_transport/compressed
```

Check available transports:

```bash
ros2 run image_transport list_transports
```
