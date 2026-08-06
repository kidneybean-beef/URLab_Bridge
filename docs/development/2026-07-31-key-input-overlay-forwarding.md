# Key Input Overlay Forwarding

URLab_Bridge now forwards the browser key map through the existing
`runtime.set_twist_control_state` RPC. The payload still includes
`dash_active` for older plugin compatibility, and adds `keys` with the eight
web controls: `w`, `s`, `q`, `e`, `a`, `d`, `space`, and `shift`.

This lets the URLab viewport overlay show the actual LAN/web command state for
the selected articulation instead of inferring state from twist velocity.
Release and stale-command paths send all keys false so the overlay clears when
control stops.
