use circuit::angle_is_zero;
use circuit::CircuitSeq;
use circuit::Gate;

use std::collections::HashMap;
use std::hash::RandomState;

// Parity of a wire, Zobrist-hashed: lin is the xor of the random keys of the variables in the
// parity, and neg is set when the parity is complemented (an odd number of X gates)
#[derive(PartialEq, Eq, Hash, Clone, Copy, Debug)]
struct WireState {
    lin: u128,
    neg: bool,
}

// Fixed-seed splitmix64 stream giving each variable (input wire or H output) its random key
struct VariableKeys {
    seed: u64,
}

impl VariableKeys {

    fn new() -> Self {
        Self{ seed: 0x2545_F491_4F6C_DD1D }
    }

    fn next_u64(&mut self) -> u64 {
        self.seed = self.seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    fn next_key(&mut self) -> u128 {
        ((self.next_u64() as u128) << 64) | (self.next_u64() as u128)
    }

}

impl WireState {

    // Create new state for a new variable (the start of a wire, or the output of an H gate)
    fn new_variable(keys: &mut VariableKeys) -> Self {
        Self{ lin: keys.next_key(), neg: false }
    }

    // Create new state for CX action on qubit
    fn xor(first: &WireState, second: &WireState) -> Self {
        Self{ lin: first.lin ^ second.lin, neg: first.neg ^ second.neg }
    }

    // Create new state for X action on qubit: X maps parity p to 1 xor p
    fn flip_all(current_state: &WireState) -> Self {
        Self{ lin: current_state.lin, neg: !current_state.neg }
    }

}

pub fn merge_rotations(c: &mut CircuitSeq, push_back: bool) {

    //Map binary states to (location, rotation, wire)
    let mut rotation_merge_dict: HashMap<WireState, (usize, f64, usize), RandomState> = HashMap::new();

    let mut keys = VariableKeys::new();

    //Map wire to state
    let mut state_dict: Vec<WireState> = Vec::with_capacity(c.num_qubits);

    for wire in 0..c.num_qubits {
        //check if qubit has been pruned out
        //TODO: Do in one line?:
        let state: WireState = WireState::new_variable(&mut keys);
        state_dict.push(state);
        //TODO: Check this:
        // All rotations on initial state could be placed at start of circuit
        rotation_merge_dict.insert(state, (0, 0.0, wire));
    }

    for gate_index in 0..c.gates.len() {
        // println!("Gate index: {}", gate_index);
        // Why do we need to dereference here?
        match c.gates[gate_index] {
            // CX gates alter the state on the wire
            Gate::CX{q1: control, q2: target} => {
                state_dict[target] = WireState::xor(&state_dict[target], &state_dict[control]);
                rotation_merge_dict.entry(state_dict[target]).or_insert((gate_index + 1, 0.0, target));
            }

            // RZ gates contribute to the rotation on the current state of the wire
            Gate::RZ{param1: k, q1: q} => {
                // println!("RZ at state {:?} at index {}", state_dict[&q].clone(), gate_index);
                rotation_merge_dict.get_mut(&state_dict[q]).unwrap().1 += k;
                c.gates[gate_index] = Gate::B;
            }

            // X gate flips that bit of the state
            Gate::X(q) => {
                state_dict[q] = WireState::flip_all(&state_dict[q]);
                rotation_merge_dict.entry(state_dict[q]).or_insert((gate_index + 1, 0.0, q));
            },

            //When we encounter a Hadamard, treat it as a new virtual qubit
            Gate::H(q) => {
                let new_state: WireState = WireState::new_variable(&mut keys);
                state_dict[q] = new_state;
                rotation_merge_dict.insert(new_state, (gate_index + 1, 0.0, q));
            },
            _ => {
                // println!("Gate: {:?}", c.gates[*gate_index]);
            }
        }
        if push_back{
            for wire in 0..c.num_qubits {
                let entry = rotation_merge_dict.get_mut(&state_dict[wire]).unwrap();
                entry.0 = gate_index + 1;
                entry.2 = wire;
            }
        }
    }
    // println!("dict: {:?}", rotation_merge_dict);

    let mut rotations_to_add: Vec<(usize, f64, usize)> =
        rotation_merge_dict.values().cloned().collect();

    // sort by reversed order in gates, then by reversed wire (and angle), so the output
    // never depends on the HashMap iteration order
    rotations_to_add.sort_by(|a, b| {
        (b.0).cmp(&a.0)
            .then((b.2).cmp(&a.2))
            .then((b.1.to_bits()).cmp(&a.1.to_bits()))
    });

    // Rebuild the gates in one pass: walking the sorted rotations backwards gives the rotations
    // at each location in the order that inserting them one by one at that location leaves them
    let mut merged_gates: Vec<Gate> = Vec::with_capacity(c.gates.len() + rotations_to_add.len());
    let mut rotations = rotations_to_add.iter().rev().peekable();
    for gate_index in 0..=c.gates.len() {
        while let Some(rotation) = rotations.next_if(|rotation| rotation.0 == gate_index) {
            if !angle_is_zero(rotation.1) {
                merged_gates.push(Gate::RZ {
                    param1: rotation.1,
                    q1: rotation.2,
                });
            }
        }
        if gate_index < c.gates.len() {
            merged_gates.push(c.gates[gate_index].clone());
        }
    }
    c.gates = merged_gates;

    c.remove_identities();

}
