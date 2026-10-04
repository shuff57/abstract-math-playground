use math_playground_lib::{run, run_native, NativeArgs};

const USAGE: &str = "usage: math_playground_bin [EXPR...] [--mode 1d|2d|3d] [--dark] [--hash V1.. ] [--doc FILE.json] [--angle deg|rad] [--demo]";

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();
    let mut args = NativeArgs::default();
    let mut it = std::env::args().skip(1);
    let mut demo = false;
    while let Some(a) = it.next() {
        let mut value = |name: &str| {
            it.next().unwrap_or_else(|| {
                eprintln!("{name} needs a value\n{USAGE}");
                std::process::exit(2)
            })
        };
        match a.as_str() {
            "--demo" => demo = true,
            "--dark" => args.dark = true,
            "--mode" => args.mode = Some(value("--mode")),
            "--hash" => args.hash = Some(value("--hash")),
            "--doc" => args.doc_file = Some(value("--doc")),
            "--angle" => args.angle = Some(value("--angle")),
            "-h" | "--help" => {
                println!("{USAGE}");
                return;
            }
            s if s.starts_with("--") => {
                eprintln!("unknown option {s}\n{USAGE}");
                std::process::exit(2);
            }
            _ => args.exprs.push(a),
        }
    }
    if demo {
        pollster::block_on(run());
    } else {
        run_native(args);
    }
}
