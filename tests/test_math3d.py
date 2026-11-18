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
