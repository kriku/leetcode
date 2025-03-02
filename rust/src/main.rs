use leetcode::Solution;

fn main() {
    let mut base = 2i32;
    let mut power = 0u32;
    let mut v = vec![];

    for _n in 1..=1000 {
        let mut value = base.pow(power);
        if value > 10_000 {
            base += 1;
            power = 0;
            value = base;
        }
        v.push(value);
        power += 1;
    }

    println!("{}", Solution::tuple_same_product(v));
}
