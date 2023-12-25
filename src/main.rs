use std::{array, time::SystemTime};

use matty::Vector;

fn main() {
    const COUNT: usize = 100000;
    const WIDTH: usize = 3;

    let arr1: [Vector<i32, WIDTH>; COUNT] =
        array::from_fn(|_| array::from_fn(|_| rand::random::<i32>() % 20 - 10).into());

    let start = SystemTime::now();
    let sum: Vector<i32, WIDTH> = arr1
        .into_iter()
        .map(|v| v.resize::<4>())
        .sum::<Vector<i32, 4>>()
        .resize::<3>();
    println!(
        "Took: {}",
        SystemTime::now().duration_since(start).unwrap().as_micros()
    );
    println!("{:?}", sum);
}
