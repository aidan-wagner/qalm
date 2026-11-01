use crate::optimization::routines::{
    opt_routine_0, opt_routine_1, opt_routine_2, opt_routine_3, opt_routine_4, print_statistics,
};

use circuit::CircuitSeq;
use std::time::SystemTime;

use std::ffi::CStr;
use std::ffi::CString;
use std::os::raw::c_char;
use circuit::{parse_program, circuit_to_string, write_program};

#[no_mangle]
pub extern "C" fn run_roqc(circuit_string: *const c_char) -> *const c_char {
    // println!("Running ROQC");
    unsafe {
        let raw_string = CStr::from_ptr(circuit_string);
        if let Ok(circuit) = raw_string.to_str() {
            //println!("Old circuit is: {}", circuit);
            let return_val = optimize_from_quartz(circuit.to_string());
            //println!("new circuit is: {}", return_val);
            let c_string_result = CString::new(return_val).expect("CString conversion failed");
            c_string_result.into_raw()
        }
        else {
            let failed_result = CString::new("invalid").expect("CString conversion failed");
            failed_result.into_raw()
        }
    }
}

#[no_mangle]
pub extern "C" fn run_roqc_two_way_rotation_merge(circuit_string: *const c_char) -> *const c_char {
    // println!("Running ROQC");
    unsafe {
        let raw_string = CStr::from_ptr(circuit_string);
        if let Ok(circuit) = raw_string.to_str() {
            //println!("Old circuit is: {}", circuit);
            let return_val = optimize_from_quartz_two_way_rotation_merge(circuit.to_string());
            //println!("new circuit is: {}", return_val);
            let c_string_result = CString::new(return_val).expect("CString conversion failed");
            c_string_result.into_raw()
        }
        else {
            let failed_result = CString::new("invalid").expect("CString conversion failed");
            failed_result.into_raw()
        }
    }
}

#[no_mangle]
pub extern "C" fn run_roqc_internal(circuit_string: *const c_char) -> *const c_char {
    // println!("Running ROQC");
    unsafe {
        let raw_string = CStr::from_ptr(circuit_string);
        if let Ok(circuit) = raw_string.to_str() {
            //println!("Old circuit is: {}", circuit);
            let return_val = optimize_from_quartz_internal(circuit.to_string());
            //println!("new circuit is: {}", return_val);
            let c_string_result = CString::new(return_val).expect("CString conversion failed");
            c_string_result.into_raw()
        }
        else {
            let failed_result = CString::new("invalid").expect("CString conversion failed");
            failed_result.into_raw()
        }
    }
}

// Frees a string returned by one of the run_roqc functions
#[no_mangle]
pub extern "C" fn roqc_free_string(s: *mut c_char) {
    if s.is_null() {
        return;
    }
    unsafe {
        drop(CString::from_raw(s));
    }
}

fn optimize_from_quartz(circuit_string: String) -> String {
    let mut circuit: CircuitSeq = parse_program(&circuit_string);
    // NOTE: Change third argument to true to get verbose output
    optimize(&mut circuit, None, None);
    circuit_to_string(&circuit.gates, circuit.num_qubits)
}

fn optimize_from_quartz_two_way_rotation_merge(circuit_string: String) -> String {
    let mut circuit: CircuitSeq = parse_program(&circuit_string);
    // println!("parsed circuit gates: {:?}", circuit.gates);
    // NOTE: Change third argument to true to get verbose output
    optimize(&mut circuit, Some(&[0,1,3,2,3,1,2,4,3,4,3,2,3]), None);
    circuit_to_string(&circuit.gates, circuit.num_qubits)
}

fn optimize_from_quartz_internal(circuit_string: String) -> String {
    let mut circuit: CircuitSeq = parse_program(&circuit_string);
    // NOTE: Change third argument to true to get verbose output
    optimize(&mut circuit, Some(&[4,3,4,3]), None);
    circuit_to_string(&circuit.gates, circuit.num_qubits)
}

#[allow(dead_code)]
pub enum OptimizationType {
    Light,
    Heavy,
    Voqc,
}

// pub fn optimize_light(c: &mut CircuitSeq) -> isize {
//     // let order = [1,3,2,3,1,2,4,3,2];
//     let order = [0, 2, 1, 2, 0, 1, 3, 2, 1];
//     optimize(c, &order, false)
// }
// 
// pub fn optimize_heavy(c: &mut CircuitSeq) -> isize {
//     // let order = [1,3,2,3,1,2,5];
//     let order = [0, 2, 1, 2, 0, 1, 4];
//     optimize(c, &order, false)
// }
// 
// pub fn optimize_voqc(c: &mut CircuitSeq) -> isize {
//     // let order = [1,3,2,3,1,2,5];
//     let order = [0,1,3,2,3,1,2,4,3,2,3];
//     optimize(c, &order, false)
// }

fn print_differing_subcircuits(c_prev: &CircuitSeq, c_opt: &CircuitSeq) {

    if c_prev.gates.len() == 0 && c_opt.gates.len() == 0 {
        println!("Current and previous circuits are empty!");
        return;
    }
    else if c_prev.gates.len() == 0 {
        println!("The new circuit is empty! Printing the previous circuit:");
        for gate in &c_opt.gates {
            println!("{}", gate);
        }
        return;
    }
    else if c_opt.gates.len() == 0 {
        assert!(false);
    }

    // NOTE: These signify the first/last elements of the differeing subcircuit (inclusive)
    let mut prefix_end: usize = 0;

    let mut suffix_start_prev: usize = c_prev.gates.len() - 1;
    let mut suffix_start_opt: usize = c_opt.gates.len() - 1;

    while prefix_end < std::cmp::min(c_prev.gates.len(), c_opt.gates.len()) {
        if c_prev.gates[prefix_end] != c_opt.gates[prefix_end] {
            break;
        }
        else {
            prefix_end += 1;
        }
    }

    loop {
        if c_prev.gates[suffix_start_prev] != c_opt.gates[suffix_start_opt] {
            break;
        }
        else if suffix_start_prev == 0 || suffix_start_opt == 0 {
            break;
        }
        else {
            suffix_start_prev -= 1;
            suffix_start_opt -= 1;
        }

    }

    if prefix_end < std::cmp::min(c_prev.gates.len(), c_opt.gates.len()) {
        println!("Printing differing subcircuit sections after last optimization routine");
        println!("Previous Circuit index ({prefix_end} to index {suffix_start_prev}) and Optimized Circuit (({prefix_end} to index {suffix_start_opt}):");

    }
    let mut circuit_index = prefix_end;
    let mut good_formatting = true;

    while circuit_index <= std::cmp::max(suffix_start_prev, suffix_start_opt) {
        let mut line: String = "".to_string();
        if circuit_index <= suffix_start_prev {
            if c_prev.gates[circuit_index].to_string().len() > 50 {
                if good_formatting {
                    println!("Gate strings are too long for good print formatting");
                }
                good_formatting = false;
            }
            line +=  &c_prev.gates[circuit_index].to_string();
            if good_formatting {
                for _ in 0..(40 - c_prev.gates[circuit_index].to_string().len()){
                    line += " ";
                }
            }
        }
        else {
            if good_formatting {
                for _ in 0..40 {
                    line += " ";
                }
            }
        }

        if circuit_index <= suffix_start_opt {
            line +=  &c_opt.gates[circuit_index].to_string();
        }
        println!("{}", line);
        circuit_index += 1
    }
}

pub fn optimize(c: &mut CircuitSeq, order: Option<&[u8]>, verbose: Option<bool>) -> isize {
    // Convert all Z gates into Rz
    let opt_order = order.unwrap_or(&[0,1,3,2,3,1,2,4,3,2,3]);
    let mut push_back = false;
    let print_details = verbose.unwrap_or(false);
    c.replace_z_gates();
    let initial_len: isize = c.gates.len().try_into().unwrap();
    let mut step_number: usize = 0;
    let mut c_prev: CircuitSeq = CircuitSeq::new(vec![], 0);
    if print_details {
        c_prev = c.clone();
    }
    loop {
        for routine in opt_order {
            step_number += 1;
            if print_details {
                println!("--- Running Routine #{routine}");
            }
            let now = SystemTime::now();
            match routine {
                0 => opt_routine_0(c),
                1 => opt_routine_1(c),
                2 => opt_routine_2(c),
                3 => opt_routine_3(c),
                4 => opt_routine_4(c, &mut push_back), // TODO: Find Error!!
                _ => continue,
            }
            c.remove_identities();
            if print_details {
                println!("took {} seconds", now.elapsed().unwrap().as_secs_f32());
                println!("New Length is: {}", c.gates.len());
                let filename: String = format!("./circuit_step_{step_number}.qasm");
                write_program(&c.gates, c.num_qubits, filename).expect("Unable to write optimized circuit to file");

            }
            // println!("Circuit is {:?}", c.gates);
        }
        if print_details {
            print_differing_subcircuits(&c_prev, &c);
            unsafe {
                print_statistics();
            }
            c.print_gate_counts();
        }
        /// println!("Circuit: {:?}", c.gates);
        // c.reduce_angles();
        return initial_len - <usize as TryInto<isize>>::try_into(c.gates.len()).unwrap();
    }
}
