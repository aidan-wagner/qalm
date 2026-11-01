pub type QubitIndex = usize;
pub type Real = f64;
pub type GateIndex = usize;

// Angles closer than this (mod 2pi) are treated as equal
const ANGLE_EPS: Real = 1e-9;

// Reduce an angle to [0, 2pi), snapping values within ANGLE_EPS below 2pi to 0
pub fn norm_angle(theta: Real) -> Real {
    let two_pi = 2.0 * std::f64::consts::PI;
    let angle = theta.rem_euclid(two_pi);
    if two_pi - angle < ANGLE_EPS {
        0.0
    } else {
        angle
    }
}

pub fn angles_equal(a: Real, b: Real) -> bool {
    norm_angle(a - b) < ANGLE_EPS
}

pub fn angle_is_zero(theta: Real) -> bool {
    norm_angle(theta) < ANGLE_EPS
}
