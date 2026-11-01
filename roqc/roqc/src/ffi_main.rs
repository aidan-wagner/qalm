use std::env;
use std::ffi::{CStr, CString};
use std::fs;
use std::io::{BufWriter, Write};
use std::os::raw::c_char;
use std::time::Instant;

mod optimization;
use optimization::optimizer::{roqc_free_string, run_roqc, run_roqc_two_way_rotation_merge};

// Calls an ROQC entry point the way the C++ side does and prints each result followed by a
// "//END" line; per-call times go to stderr
// Usage: roqc_ffi <a|b> <qasm file>...   (a = run_roqc, b = run_roqc_two_way_rotation_merge)
fn main() {
    let args: Vec<String> = env::args().collect();

    let entry: extern "C" fn(*const c_char) -> *const c_char = match args[1].as_str() {
        "a" => run_roqc,
        "b" => run_roqc_two_way_rotation_merge,
        _ => panic!("Unknown entry point: {}", args[1]),
    };

    let mut out = BufWriter::new(std::io::stdout().lock());
    for filename in &args[2..] {
        let circuit_from_file = fs::read_to_string(filename).expect("Error reading circuit file");
        let circuit_string = CString::new(circuit_from_file).expect("CString conversion failed");

        let now = Instant::now();
        let raw_result = entry(circuit_string.as_ptr());
        let result = unsafe { CStr::from_ptr(raw_result) }
            .to_str()
            .expect("ROQC returned invalid UTF-8")
            .to_string();
        roqc_free_string(raw_result as *mut c_char);
        let elapsed = now.elapsed().as_secs_f64();

        out.write_all(result.as_bytes()).unwrap();
        if !result.ends_with('\n') {
            out.write_all(b"\n").unwrap();
        }
        out.write_all(b"//END\n").unwrap();
        out.flush().unwrap();
        eprintln!("{}\t{:.6}", filename, elapsed);
    }
}
