from toshot.math3d import (
    euler_ypr_to_quat,
    quat_to_euler_ypr,
    stabilization_rotation,
)
from toshot.source.models import AttitudeSample, StabMode
from toshot.math3d import frame_rotations


def test_quat_euler_roundtrip():
    q = euler_ypr_to_quat(30, -10, 5)
    yaw, pitch, roll = quat_to_euler_ypr(q)
    assert abs(yaw - 30) < 1e-3
    assert abs(pitch + 10) < 1e-3 or abs(pitch - (-10)) < 1e-3
    assert abs(roll - 5) < 1e-3


def test_horizon_zeros_pitch_roll_keeps_yaw_follow():
    q = euler_ypr_to_quat(40, 12, -8)
    yaw, pitch, roll = stabilization_rotation(q, StabMode.HORIZON)
    # Follow yaw (no cancel), cancel pitch/roll
    assert abs(yaw) < 1e-6
    assert abs(pitch - (-12)) < 1e-3
    assert abs(roll - 8) < 1e-3


def test_frame_rotations_length():
    samples = [
        AttitudeSample(0.0, *euler_ypr_to_quat(0, 0, 0)),
        AttitudeSample(1.0, *euler_ypr_to_quat(10, 5, 0)),
    ]
    rots = frame_rotations(samples, fps=10, n_frames=10, mode=StabMode.WORLD)
    assert len(rots) == 10


# Avata 360 in flight (lenses up/down) and a lens-calibration quaternion that
# DJI interleaves into the djmd attitude stream.
_AVATA_Q = (0.094, -0.694, 0.110, 0.705)
_CALIB_Q = (0.002, -0.001, 0.0, -1.0)


def test_stabilizer_puts_gravity_down_for_vertical_lenses():
    import numpy as np

    from toshot.math3d import Stabilizer

    samples = [AttitudeSample(t * 0.01, *_AVATA_Q) for t in range(20)]
    s = Stabilizer(samples, StabMode.HORIZON).matrix(0.1)
    # Output "down" must land on the lens looking at the ground (stitch +z)
    down_in_stitch = s @ np.array([0.0, -1.0, 0.0])
    assert down_in_stitch[2] > 0.99
    assert np.allclose(s @ s.T, np.eye(3), atol=1e-6)


def test_attitude_track_rejects_interleaved_calibration_quat():
    from toshot.math3d import AttitudeTrack

    samples = [AttitudeSample(t * 0.001, *(_CALIB_Q if t % 7 == 0 else _AVATA_Q)) for t in range(70)]
    q = AttitudeTrack(samples).quat_at(0.035)
    assert abs(sum(a * b for a, b in zip(q, _AVATA_Q))) > 0.999
