use criterion::{black_box, criterion_group, criterion_main, Criterion};
use maybenot_gen::random::random_machine;
use rand_xoshiro::rand_core::SeedableRng;
use rand_xoshiro::Xoshiro256StarStar;

pub fn random_machine_rng_source_benchmarks(c: &mut Criterion) {
    let n = 1000;

    c.bench_function("1000 machines with 4 states, thread_rng()", |b| {
        let rng = &mut rand::thread_rng();
        b.iter(|| {
            gen_machine(rng, 4, black_box(n));
        })
    });
    c.bench_function("1000 machines with 4 states, Xoshiro256StarStar", |b| {
        let rng = &mut Xoshiro256StarStar::seed_from_u64(0);
        b.iter(|| {
            gen_machine(rng, 4, black_box(n));
        })
    });
    c.bench_function("1000 machines with 6 states, thread_rng()", |b| {
        let rng = &mut rand::thread_rng();
        b.iter(|| {
            gen_machine(rng, 6, black_box(n));
        })
    });
    c.bench_function("1000 machines with 6 states, Xoshiro256StarStar", |b| {
        let rng = &mut Xoshiro256StarStar::seed_from_u64(0);
        b.iter(|| {
            gen_machine(rng, 6, black_box(n));
        })
    });
}

criterion_group!(benches, random_machine_rng_source_benchmarks);
criterion_main!(benches);

fn gen_machine<R: rand::Rng>(rng: &mut R, num_states: usize, n: usize) {
    for _ in 0..n {
        random_machine(
            num_states, false, false, false, false, false, None, None, None, rng,
        );
    }
}
