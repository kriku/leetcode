use std::cell::RefCell;
use std::cmp::Reverse;
use std::cmp::{self, Ordering};
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap, HashSet, VecDeque};
use std::i32;
use std::rc::Rc;
use std::str;

pub struct Solution {}

/**
 * https://leetcode.com/problems/special-array-i/
 */
impl Solution {
    pub fn is_array_special(nums: Vec<i32>) -> bool {
        let is_even = |n| n % 2 == 0;
        let mut parity = is_even(nums[0]);

        for n in nums {
            if is_even(n) != parity {
                return false;
            }
            parity = !parity;
        }

        return true;
    }

    pub fn is_array_special_short(nums: Vec<i32>) -> bool {
        !nums.as_slice().windows(2).any(|x| (x[0] + x[1]) % 2 == 0)
    }
}

#[cfg(test)]
mod is_array_special {
    use super::*;

    #[test]
    fn case_1() {
        let result = Solution::is_array_special(vec![2, 1, 6]);
        assert_eq!(result, true);
    }
}

/**
 * https://leetcode.com/problems/check-if-array-is-sorted-and-rotated/
 */
impl Solution {
    pub fn check(nums: Vec<i32>) -> bool {
        let mut x = nums.len();

        for i in 0..nums.len() - 1 {
            if nums[i] > nums[i + 1] {
                x = i + 1;
                break;
            }
        }

        for i in 0..nums.len() - 1 {
            let j = (i + x) % nums.len();
            let k = (i + x + 1) % nums.len();

            if nums[j] > nums[k] {
                return false;
            }
        }

        return true;
    }
}

#[cfg(test)]
mod check {
    use super::*;

    #[test]
    fn case_1() {
        let result = Solution::check(vec![3, 4, 5, 1, 2]);
        assert_eq!(result, true);
    }

    #[test]
    fn case_2() {
        let result = Solution::check(vec![2, 1, 3, 4]);
        assert_eq!(result, false);
    }

    #[test]
    fn case_3() {
        let result = Solution::check(vec![1, 2, 3]);
        assert_eq!(result, true);
    }

    #[test]
    fn case_4() {
        let result = Solution::check(vec![1, 1, 1]);
        assert_eq!(result, true);
    }

    #[test]
    fn case_5() {
        let result = Solution::check(vec![2, 1]);
        assert_eq!(result, true);
    }
}

/**
 * https://leetcode.com/problems/maximum-ascending-subarray-sum/
 */
impl Solution {
    pub fn max_ascending_sum(nums: Vec<i32>) -> i32 {
        let mut sum = nums[0];
        let mut max_sum = sum;

        for (previous, next) in nums.iter().zip(nums.iter().skip(1)) {
            match next.cmp(previous) {
                Ordering::Equal | Ordering::Less => {
                    sum = *next;
                }
                Ordering::Greater => {
                    sum += next;
                }
            }

            max_sum = cmp::max(max_sum, sum);
        }

        return max_sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn case_1() {
        let result = Solution::max_ascending_sum(vec![10, 20, 30, 5, 10, 50]);
        assert_eq!(result, 65);
    }

    #[test]
    fn case_2() {
        let result = Solution::max_ascending_sum(vec![10]);
        assert_eq!(result, 10);
    }
}

/**
 * https://leetcode.com/problems/check-if-one-string-swap-can-make-strings-equal/
 */
impl Solution {
    pub fn are_almost_equal(s1: String, s2: String) -> bool {
        let mut d = vec![];
        for (c1, c2) in s1.chars().zip(s2.chars()) {
            if c1 != c2 {
                d.push((c1, c2));
            }
        }

        if d.len() == 0 {
            return true;
        }

        if d.len() != 2 {
            return false;
        }

        if d[0].0 == d[1].1 && d[1].0 == d[0].1 {
            return true;
        }

        return false;
    }
}

#[cfg(test)]
mod tests_are_almost_equal {
    use super::*;

    #[test]
    fn case_1() {
        let result = Solution::are_almost_equal("bank".to_string(), "kanb".to_string());
        assert_eq!(result, true);
    }

    #[test]
    fn case_2() {
        let result = Solution::are_almost_equal("attack".to_string(), "defend".to_string());
        assert_eq!(result, false);
    }

    #[test]
    fn case_3() {
        let result = Solution::are_almost_equal("kelb".to_string(), "kelb".to_string());
        assert_eq!(result, true);
    }
}

/**
 * https://leetcode.com/problems/tuple-with-same-product/
 */
impl Solution {
    pub fn product_pairs_count(nums: &Vec<i32>) -> BTreeMap<i32, i32> {
        let mut products: BTreeMap<i32, i32> = BTreeMap::new();

        // 1000 (10^3)
        for i in 0..nums.len() {
            for j in i + 1..nums.len() {
                if i != j {
                    let product = nums[i] * nums[j];
                    let count = products.entry(product).or_insert(0);
                    *count += 1;
                }
            }
        }

        return products;
    }

    pub fn product_pairs(nums: Vec<i32>) -> HashMap<i32, Vec<(i32, i32)>> {
        let mut products = HashMap::new();

        for i in 0..nums.len() {
            for j in i + 1..nums.len() {
                if i != j {
                    let product = nums[i] * nums[j];
                    let pairs = products.entry(product).or_insert(vec![]);
                    pairs.push((nums[i], nums[j]));
                }
            }
        }

        return products;
    }

    pub fn tuple_same_product(nums: Vec<i32>) -> i32 {
        let mut products: HashMap<i32, i32> = HashMap::new();

        for i in 0..nums.len() {
            for j in i + 1..nums.len() {
                if i != j {
                    let product = nums[i] * nums[j];
                    let count = products.entry(product).or_insert(0);
                    *count += 1;
                }
            }
        }

        let mut count = 0;

        for pairs in products.into_values() {
            if pairs > 1 {
                count += 4 * pairs * (pairs - 1);
            }
        }

        return count;
    }
}

#[cfg(test)]
mod tuple_same_product {
    use super::*;

    #[test]
    fn case_1() {
        let result = Solution::tuple_same_product(vec![2, 3, 4, 6]);
        assert_eq!(result, 8);
    }

    #[test]
    fn case_2() {
        let result = Solution::tuple_same_product(vec![1, 2, 4, 5, 10]);
        assert_eq!(result, 16);
    }

    #[test]
    fn case_3() {
        /*
         * 24: (2, 12), (3, 8), (4, 6) = 8 + 8 + 8 = 24  (2^(p-1) - 1)*8
         * 48: (4, 12), (6, 8) = 8
         * 12: (2, 6), (3, 4) = 8
         */
        let result = Solution::tuple_same_product(vec![2, 3, 4, 6, 8, 12]);
        assert_eq!(result, 40);
    }

    #[test]
    fn case_4() {
        let result = Solution::tuple_same_product(vec![2, 3, 5, 7]);
        assert_eq!(result, 0);
    }

    #[test]
    fn case_5() {
        /*

        8: (1, 8), (2, 4)
        16: ()

         */
        let result = Solution::tuple_same_product(vec![
            1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,
        ]);
        assert_eq!(result, 1288);
    }

    #[test]
    fn case_6() {
        let result = Solution::tuple_same_product(vec![
            8, 448, 264, 525, 435, 486, 378, 308, 144, 75, 196, 110, 231, 120, 39, 288, 50, 616,
            140, 261, 272, 783, 225, 552, 598, 30, 128, 570, 322, 77, 340, 19, 72, 224, 294, 390,
            276, 87, 238, 180, 80, 33, 68, 210, 725, 243, 696, 198, 208, 46, 21, 58, 360, 170, 190,
            510, 375, 551, 348, 396, 377, 69, 84, 300, 572, 468, 160, 24, 34, 667, 29, 64, 253,
            115, 690, 100, 870, 754, 102, 1, 11, 312, 609, 161, 493, 450, 342, 133, 588, 48, 152,
            10, 42, 273, 440, 728, 65, 98, 5, 23, 250, 242, 38, 182, 26, 648, 99, 357, 400, 275,
            187, 483, 414, 323, 408, 105, 230, 520, 750, 4, 500, 32, 286, 418, 189, 638, 528, 234,
            315, 96, 352, 812, 232, 40, 3, 130, 184, 17, 15, 324, 240, 392, 7, 174, 270, 416, 513,
            25, 203, 221, 399, 475, 9, 54, 476, 442, 406, 840, 12, 504, 114, 675, 624, 621, 56,
            405, 125, 119, 136, 506, 702, 364, 70, 60, 228, 20, 85, 575, 135, 117, 78, 171, 156,
            55, 299, 462, 116, 780, 52, 432, 165, 88, 325, 338, 391, 546, 522, 209, 176, 108,
        ]);
        assert_eq!(result, 251424);
    }
}

/**
 * https://leetcode.com/problems/find-the-number-of-distinct-colors-among-the-balls/
 */
// use std::collections::HashMap;

impl Solution {
    /*
    1 <= limit <= 10^9
    1 <= n == queries.length <= 10^5
    queries[i].length == 2
    0 <= queries[i][0] <= limit
    1 <= queries[i][1] <= 10^9
    */
    pub fn query_results(_limit: i32, queries: Vec<Vec<i32>>) -> Vec<i32> {
        let mut ball_colors: HashMap<i32, i32> = HashMap::new();
        let mut color_balls: HashMap<i32, i32> = HashMap::new();
        let mut distinct_color_count = 0;
        let mut distinct_color = vec![];

        for query in queries {
            let index = query[0];
            let color = query[1];

            if let Some(current_color) = ball_colors.get_mut(&index) {
                if let Some(color_ball_count) = color_balls.get_mut(current_color) {
                    *color_ball_count -= 1;

                    if *color_ball_count == 0 {
                        distinct_color_count -= 1;
                    }
                }
            }

            ball_colors.insert(index, color);

            if let Some(color_ball_count) = color_balls.get_mut(&color) {
                if *color_ball_count == 0 {
                    distinct_color_count += 1;
                }

                *color_ball_count += 1;
            } else {
                color_balls.insert(color, 1);
                distinct_color_count += 1;
            }

            distinct_color.push(distinct_color_count);
        }

        return distinct_color;
    }
}

#[cfg(test)]
mod query_results {
    use super::*;

    #[test]
    fn case_1() {
        let result =
            Solution::query_results(5, vec![vec![1, 1], vec![2, 1], vec![3, 1], vec![4, 1]]);
        assert_eq!(result, vec![1, 1, 1, 1]);
    }

    #[test]
    fn case_2() {
        let result = Solution::query_results(5, vec![vec![1, 3], vec![2, 4], vec![3, 5]]);
        assert_eq!(result, vec![1, 2, 3]);
    }

    #[test]
    fn case_3() {
        let result = Solution::query_results(5, vec![vec![1, 5], vec![1, 4], vec![1, 3]]);
        assert_eq!(result, vec![1, 1, 1]);
    }

    #[test]
    fn case_4() {
        let result = Solution::query_results(
            5,
            vec![vec![1, 5], vec![2, 4], vec![1, 3], vec![2, 2], vec![1, 2]],
        );
        assert_eq!(result, vec![1, 2, 2, 2, 1]);
    }

    #[test]
    fn case_5() {
        let result =
            Solution::query_results(5, vec![vec![1, 4], vec![2, 5], vec![1, 3], vec![3, 4]]);
        assert_eq!(result, vec![1, 2, 2, 3])
    }
}

/**
 * https://leetcode.com/problems/design-a-number-container-system/
 */
// use::std::collections::{HashMap, BTreeSet};

pub struct NumberContainers {
    numbers: HashMap<i32, i32>,
    positions: HashMap<i32, BTreeSet<i32>>,
}

/**
 * `&self` means the method takes an immutable reference.
 * If you need a mutable reference, change it to `&mut self` instead.
 */
impl NumberContainers {
    pub fn new() -> Self {
        NumberContainers {
            numbers: HashMap::new(),
            positions: HashMap::new(),
        }
    }

    pub fn change(&mut self, index: i32, number: i32) {
        let maybe_n = self.numbers.get(&index);

        if let Some(n) = maybe_n {
            self.positions.entry(*n).and_modify(|p| {
                p.remove(&index);
            });
        }

        self.numbers.insert(index, number);
        let ps = self.positions.entry(number).or_insert(BTreeSet::new());

        ps.insert(index);
    }

    pub fn find(&self, number: i32) -> i32 {
        let ps = self.positions.get(&number);

        match ps {
            None => -1,
            Some(p) => match p.first() {
                None => -1,
                Some(key) => *key,
            },
        }
    }
}

/**
 * Your NumberContainers object will be instantiated and called as such:
 * let obj = NumberContainers::new();
 * obj.change(index, number);
 * let ret_2: i32 = obj.find(number);
 */

#[cfg(test)]
mod number_containers {
    use super::*;

    #[test]
    fn case_1() {
        let mut nc = NumberContainers::new();
        nc.change(1, 2);
        let result = nc.find(2);
        assert_eq!(result, 1);
    }
}

/**
 * https://leetcode.com/problems/count-number-of-bad-pairs/
 */
// use std::collections::HashMap;

impl Solution {
    pub fn count_bad_pairs(nums: Vec<i32>) -> i64 {
        let n = nums.len();
        let mut pairs = (n * (n - 1) / 2) as u64;

        let mut pair_with_diff = HashMap::new();

        for (index, num) in nums.iter().enumerate() {
            let diff = num - index as i32;

            let p = pair_with_diff.entry(diff).or_insert(0);
            *p += 1usize;
        }

        for v in pair_with_diff.into_values() {
            pairs -= (v * (v - 1) / 2) as u64;
        }

        return pairs as i64;
    }
}

#[cfg(test)]
mod count_bad_pairs {
    use super::*;

    /**
     * good pairs
     * (0, 2), (1, 3)
     */
    #[test]
    fn case_1() {
        let result = Solution::count_bad_pairs(vec![1, 2, 3, 4, 5]);
        assert_eq!(result, 0);
    }

    #[test]
    fn case_2() {
        let result = Solution::count_bad_pairs(vec![4, 1, 3, 3]);
        assert_eq!(result, 5);
    }
}

/**
 * https://leetcode.com/problems/clear-digits/
 */
impl Solution {
    pub fn clear_digits(s: String) -> String {
        let mut result = String::new();
        for c in s.chars() {
            if c >= '0' && c <= '9' {
                result.pop();
            } else {
                result.push(c);
            }
        }
        return result;
    }
}

#[cfg(test)]
mod clear_digits {
    use super::*;

    #[test]
    fn case_1() {
        let result = Solution::clear_digits(String::from("abc"));
        assert_eq!(result, "abc");
    }

    #[test]
    fn case_2() {
        let result = Solution::clear_digits(String::from("cd34"));
        assert_eq!(result, "");
    }
}

/**
 * https://leetcode.com/problems/remove-all-occurrences-of-a-substring/
 */

impl Solution {
    pub fn remove_occurrences(s: String, part: String) -> String {
        let pattern: Vec<char> = part.chars().collect();
        let mut prefixes = Vec::<Vec<usize>>::new();
        let mut result: Vec<char> = s.chars().collect();
        let mut deleted = 0;

        for (index, c) in s.chars().enumerate() {
            let mut next_prefixes = vec![];

            if c == pattern[0] {
                next_prefixes.push(1);
            }

            if let Some(previous_prefixes) = prefixes.last() {
                for prefix in previous_prefixes {
                    if *prefix < pattern.len() && c == pattern[*prefix] {
                        next_prefixes.push(*prefix + 1);
                    }
                }
            }

            let matched = next_prefixes
                .iter()
                .find(|&p| *p == pattern.len())
                .is_some();

            if matched {
                prefixes = prefixes[..prefixes.len() + 1 - pattern.len()].to_vec();
                let remove_from = index + 1 - deleted - pattern.len();
                let remove_to = index + 1 - deleted;
                let _: Vec<char> = result.splice(remove_from..remove_to, []).collect();
                deleted += pattern.len();
                continue;
            }

            prefixes.push(next_prefixes);
        }

        return result.into_iter().collect();
    }
}

#[cfg(test)]
mod remove_occurrences {
    use super::*;

    #[test]
    fn case_1() {
        let result = Solution::remove_occurrences(String::from("daabc"), String::from("abc"));
        assert_eq!(result, "da");
    }

    #[test]
    fn case_2() {
        let result = Solution::remove_occurrences(String::from("axxxyyyyb"), String::from("xy"));
        assert_eq!(result, "ayb");
    }

    #[test]
    fn case_3() {
        let result = Solution::remove_occurrences(String::from("sshvahvaln"), String::from("hva"));
        assert_eq!(result, "ssln");
    }

    #[test]
    fn case_4() {
        let result = Solution::remove_occurrences(
            String::from("qtbxqtbxelkekgcdnelkeqtbxelkekgcdnqtbxelkekgcdnkgcdnwqchzunbvyjoq"),
            String::from("qtbxelkekgcdn"),
        );
        assert_eq!(result, "wqchzunbvyjoq");
    }

    #[test]
    fn case_5() {
        let result = Solution::remove_occurrences(
            String::from("wwwwwwwwwwwwwwwwwwwwwvwwwwswxwwwwsdwxweeohapwwzwuwajrnogb"),
            String::from("w"),
        );
        assert_eq!(result, "vsxsdxeeohapzuajrnogb");
    }

    #[test]
    fn case_6() {
        let result = Solution::remove_occurrences(String::from("aaababc"), String::from("aab"));
        assert_eq!(result, "c");
    }
}

/**
 * https://leetcode.com/problems/max-sum-of-a-pair-with-equal-sum-of-digits/
 */
// use std::collections::{HashMap, BinaryHeap};

fn num_sum(mut num: i32) -> i32 {
    let mut sum = 0;

    while num >= 10 {
        sum += num % 10;
        num = num / 10;
    }

    return sum + num;
}

impl Solution {
    pub fn maximum_sum(nums: Vec<i32>) -> i32 {
        let mut num_sums = HashMap::new();

        for num in nums {
            let sum = num_sum(num);
            let ns = num_sums.entry(sum).or_insert(BinaryHeap::new());
            ns.push(num);
        }

        let mut max = -1;

        for ns in num_sums.values_mut() {
            let mut sum = || Some(ns.pop()? + ns.pop()?);

            if let Some(s) = sum() {
                max = max.max(s);
            }
        }

        return max;
    }
}

#[cfg(test)]
mod maximum_sum {
    use super::*;

    #[test]
    fn case_0() {
        let result = Solution::maximum_sum(vec![
            229, 398, 269, 317, 420, 464, 491, 218, 439, 153, 482, 169, 411, 93, 147, 50, 347, 210,
            251, 366, 401,
        ]);
        assert_eq!(result, 973);
    }

    #[test]
    fn case_1() {
        let result = Solution::maximum_sum(vec![18, 43, 36, 13, 7]);
        assert_eq!(result, 54);
    }

    #[test]
    fn case_2() {
        let result = Solution::maximum_sum(vec![10, 12, 19, 14]);
        assert_eq!(result, -1);
    }

    #[test]
    fn case_3() {
        let result = Solution::maximum_sum(vec![4, 6, 10, 6]);
        assert_eq!(result, 12);
    }
}

/**
 * https://leetcode.com/problems/minimum-operations-to-exceed-threshold-value-ii/
 */
// use std::cmp::Reverse;
// use std::rc::Rc;

impl Solution {
    pub fn min_operations(nums: Vec<i32>, k: i32) -> i32 {
        let mut heap = BinaryHeap::new();

        for n in nums {
            if n < k {
                heap.push(Reverse(n));
            }
        }

        let mut count = 0;

        while heap.len() > 0 {
            count += 1;

            let mut v = || heap.pop()?.0.checked_mul(2)?.checked_add(heap.pop()?.0);

            if let Some(v) = v() {
                if v < k {
                    heap.push(Reverse(v));
                }
            }
        }

        return count;
    }
}

#[cfg(test)]
mod min_operations {
    use super::*;

    #[test]
    fn case_1() {
        let result = Solution::min_operations(vec![1, 2, 3, 4, 5], 5);
        assert_eq!(result, 3);
    }

    #[test]
    fn case_2() {
        let result = Solution::min_operations(vec![2, 11, 10, 1, 3], 10);
        assert_eq!(result, 2);
    }

    #[test]
    fn case_3() {
        let result = Solution::min_operations(vec![1, 1, 2, 4, 9], 20);
        assert_eq!(result, 4);
    }

    #[test]
    fn case_4() {
        let result = Solution::min_operations(vec![999999999, 999999999, 999999999], 1000000000);
        assert_eq!(result, 2);
    }
}

/**
 * https://leetcode.com/problems/product-of-the-last-k-numbers/
 */
struct ProductOfNumbers {
    products: Vec<i32>,
}

/**
 * `&self` means the method takes an immutable reference.
 * If you need a mutable reference, change it to `&mut self` instead.
 */
impl ProductOfNumbers {
    fn new() -> Self {
        ProductOfNumbers {
            products: Vec::new(),
        }
    }

    fn add(&mut self, num: i32) {
        if num == 0 {
            self.products = Vec::new();
            return;
        }

        if let Some(prefix) = self.products.last() {
            self.products.push(*prefix * num);
            return;
        }

        self.products.push(num);
    }

    fn _get_product(&self, k: usize) -> Option<i32> {
        let p = &self.products;
        if k > p.len() {
            None
        } else if k == p.len() {
            Some(*p.last()?)
        } else {
            Some(p.last()? / p.get(p.len() - k - 1)?)
        }
    }

    fn get_product(&self, k: i32) -> i32 {
        let k: usize = k as usize;

        match self._get_product(k) {
            Some(v) => v,
            None => 0,
        }
    }
}

/**
 * Your ProductOfNumbers object will be instantiated and called as such:
 * let obj = ProductOfNumbers::new();
 * obj.add(num);
 * let ret_2: i32 = obj.get_product(k);
 */
#[cfg(test)]
mod product_of_numbers {
    use super::*;

    #[test]
    fn case_1() {
        let mut obj = ProductOfNumbers::new();
        obj.add(3);
        obj.add(0);
        obj.add(2);
        obj.add(5);
        obj.add(4);
        let result = obj.get_product(2);
        assert_eq!(result, 20);
        let result = obj.get_product(3);
        assert_eq!(result, 40);
    }
}

/**
 * https://leetcode.com/problems/find-the-punishment-number-of-an-integer/
 */
impl Solution {
    fn can_punish(n: i32) -> bool {
        let square = n * n;
        let s = square.to_string();
        let len: u32 = s.len().saturating_sub(1).try_into().unwrap();

        if n == 1 {
            return true;
        }

        for i in 1..(2_u32.pow(len)) {
            let mut sum = 0;
            let mut cursor = 0u32;
            // let mut parts = vec![];

            for j in 0..len {
                let bit = i >> j & 1;

                if bit == 1 {
                    let part = square % 10i32.pow(len - cursor + 1) / 10i32.pow(len - j);
                    // parts.push(part);
                    sum += part;
                    cursor = j + 1;
                }
            }
            let rest = square % 10i32.pow(len - cursor + 1);
            // parts.push(rest);
            sum += rest;

            if sum == n {
                // println!("{:?}", parts);
                return true;
            }
        }

        return false;
    }

    pub fn punishment_number(n: i32) -> i32 {
        let mut sum = 0;

        for i in 1..=n {
            if Self::can_punish(i) {
                // println!("{} !!!!", i);
                sum += i * i;
            }
        }

        return sum;
    }
}

#[cfg(test)]
mod punishment_number {
    use super::*;

    #[test]
    fn case_1() {
        let result = Solution::punishment_number(10);
        assert_eq!(result, 182);
    }

    #[test]
    fn case_2() {
        let result = Solution::punishment_number(37);
        assert_eq!(result, 1478);
    }
}

/**
 * https://leetcode.com/problems/construct-the-lexicographically-largest-valid-sequence/
 */

/*

0 0 0 0 0 0 0 0 0 0 ... 0
1
2 1 2
3 1 2 3 2
4 2 3 2 4 3 1

5 2 4 2 3 5 4 3 1
5 3 1 4 3 5 2 4 2

6 4 2 5 2 4 6 3 5 1 3
*/

impl Solution {
    // left - sorted descending
    // s -
    fn find_place(left: &Vec<i32>, s: &Vec<i32>) -> Option<Vec<i32>> {
        // println!("{:?} {:?}", left, s);

        if left.len() == 0 {
            return Some(s.clone());
        }

        for (place, num) in s.iter().enumerate() {
            if *num != 0 {
                continue;
            }

            for (index, n) in left.iter().enumerate() {
                let other_place = place + TryInto::<usize>::try_into(*n).unwrap();
                let new_left = [&left[..index], &left[index + 1..]].concat();

                if *n > 1 {
                    if let Some(0) = s.get(other_place) {
                        let mut new_s = s.clone();
                        new_s[place] = *n;
                        new_s[other_place] = *n;

                        let result = Self::find_place(&new_left, &new_s);

                        if result.is_some() {
                            return result;
                        }
                    }
                } else {
                    let mut new_s = s.clone();
                    new_s[place] = *n;

                    let result = Self::find_place(&new_left, &new_s);

                    if result.is_some() {
                        return result;
                    }
                }
            }
        }

        return None;
    }

    pub fn construct_distanced_sequence(n: i32) -> Vec<i32> {
        let sequence = vec![0i32; 2usize * TryInto::<usize>::try_into(n).unwrap() - 1];
        let left: Vec<i32> = (1..=n).rev().collect();

        match Self::find_place(&left, &sequence) {
            None => vec![],
            Some(r) => r,
        }
    }
}

#[cfg(test)]
mod construct_distanced_sequence {
    use super::*;

    #[test]
    fn case_1() {
        let result = Solution::construct_distanced_sequence(3);
        assert_eq!(result, vec![3, 1, 2, 3, 2]);
    }

    #[test]
    fn case_2() {
        let result = Solution::construct_distanced_sequence(5);
        assert_eq!(result, vec![5, 3, 1, 4, 3, 5, 2, 4, 2]);
    }

    #[test]
    fn case_3() {
        let result = Solution::construct_distanced_sequence(13);
        assert_eq!(
            result,
            vec![13, 11, 12, 8, 6, 4, 9, 10, 1, 4, 6, 8, 11, 13, 12, 9, 7, 10, 3, 5, 2, 3, 2, 7, 5]
        );
    }
}

/**
 * https://leetcode.com/problems/construct-smallest-number-from-di-string/
 * constructive solution
 */
impl Solution {
    pub fn smallest_number_2(pattern: String) -> String {
        let mut stack: Vec<usize> = vec![];
        let mut res: Vec<usize> = vec![];

        for (i, c) in pattern.chars().enumerate() {
            match c {
                'I' => {
                    res.push(i + 1);
                    while stack.len() > 0 {
                        res.push(stack.pop().unwrap());
                    }
                }
                _ => stack.push(i + 1),
            }
        }

        res.push(stack.len() + res.len() + 1);

        while stack.len() > 0 {
            res.push(stack.pop().unwrap());
        }

        res.into_iter().map(|i| i.to_string()).collect::<String>()
    }
}

/**
 * https://leetcode.com/problems/construct-smallest-number-from-di-string/
 */
impl Solution {
    fn backtracking(
        pattern: &Vec<char>,
        p: &mut Vec<usize>,
        used: &mut HashSet<usize>,
        i: usize,
    ) -> bool {
        if i == p.len() {
            return true;
        }

        if i == 0 {
            for j in 1..=9usize {
                used.insert(j);
                p[i] = j;

                if Self::backtracking(pattern, p, used, i + 1) {
                    return true;
                }

                used.remove(&j);
            }
        } else {
            let c = pattern[i - 1];
            for j in 1..=9usize {
                if used.contains(&j) {
                    continue;
                }

                match c {
                    'D' => {
                        if j >= p[i - 1] {
                            continue;
                        }
                    }
                    'I' => {
                        if j <= p[i - 1] {
                            continue;
                        }
                    }
                    _ => {}
                }

                used.insert(j);
                p[i] = j;

                if Self::backtracking(pattern, p, used, i + 1) {
                    return true;
                }

                used.remove(&j);
            }
        }

        return false;
    }

    pub fn smallest_number(pattern: String) -> String {
        let pattern: Vec<char> = pattern.chars().collect();
        let mut p: Vec<usize> = vec![0; pattern.len() + 1];
        let mut used: HashSet<usize> = HashSet::new();

        let _posible = Self::backtracking(&pattern, &mut p, &mut used, 0);

        return p.iter().map(|v| v.to_string()).collect();
    }
}

#[cfg(test)]
mod smallest_number {
    use super::*;

    #[test]
    fn case_1() {
        let result = Solution::smallest_number_2(String::from("IIIDIDDD"));
        assert_eq!(result, "123549876");
    }

    #[test]
    fn case_2() {
        let result = Solution::smallest_number(String::from("DDD"));
        assert_eq!(result, "4321");
    }
}

/**
 * https://leetcode.com/problems/the-k-th-lexicographical-string-of-all-happy-strings-of-length-n/
 */
impl Solution {
    fn get_left(c: char) -> Vec<char> {
        match c {
            'a' => vec!['b', 'c'],
            'b' => vec!['a', 'c'],
            'c' => vec!['a', 'b'],
            _ => vec!['a', 'b', 'c'],
        }
    }

    fn happy_string(current: &mut Vec<char>, n: usize, k: usize, i: usize, j: &mut usize) -> bool {
        if i == n && *j == k {
            return true;
        }

        if i == n {
            *j += 1;
            return false;
        }

        let left = if i == 0 {
            Self::get_left('w')
        } else {
            Self::get_left(current[i - 1])
        };

        for &c in left.iter() {
            current[i] = c;

            let result = Self::happy_string(current, n, k, i + 1, j);

            if result {
                return true;
            }
        }

        return false;
    }

    pub fn get_happy_string(n: i32, k: i32) -> String {
        let mut current = vec!['_'; n as usize];
        let mut j = 1;

        if Self::happy_string(&mut current, n as usize, k as usize, 0, &mut j) {
            return current.iter().collect();
        } else {
            return String::from("");
        }
    }
}

#[cfg(test)]
mod get_happy_string {
    use super::*;

    #[test]
    fn case_1() {
        let result = Solution::get_happy_string(1, 3);
        assert_eq!(result, String::from("c"));
    }

    #[test]
    fn case_2() {
        let result = Solution::get_happy_string(1, 4);
        assert_eq!(result, String::from(""));
    }

    #[test]
    fn case_3() {
        let result = Solution::get_happy_string(3, 9);
        assert_eq!(result, String::from("cab"));
    }
}

/**
 * https://leetcode.com/problems/find-unique-binary-string/
 */
impl Solution {
    pub fn find_different_binary_string(nums: Vec<String>) -> String {
        nums.iter()
            .enumerate()
            .map(|(i, s)| match s.chars().nth(i) {
                Some('0') => '1',
                Some('1') => '0',
                _ => '?', // this happens if string lengths is not equal
            })
            .collect()
    }
}

#[cfg(test)]
mod find_different_binary_string {
    use super::*;

    #[test]
    fn case_1() {
        let result =
            Solution::find_different_binary_string(vec![String::from("00"), String::from("10")]);
        assert_eq!(result, String::from("11"));
    }
}

/**
 * https://leetcode.com/problems/find-elements-in-a-contaminated-binary-tree/
 */

// Definition for a binary tree node.
// #[derive(Debug, PartialEq, Eq)]
// pub struct TreeNode {
//     pub val: i32,
//     pub left: Option<Rc<RefCell<TreeNode>>>,
//     pub right: Option<Rc<RefCell<TreeNode>>>,
// }

// impl TreeNode {
//     #[inline]
//     pub fn new(val: i32) -> Self {
//         TreeNode {
//             val,
//             left: None,
//             right: None,
//         }
//     }
// }
struct FindElements {
    values: HashSet<i32>,
}

/**
 * `&self` means the method takes an immutable reference.
 * If you need a mutable reference, change it to `&mut self` instead.
 */
impl FindElements {
    fn new(root: Option<Rc<RefCell<TreeNode>>>) -> Self {
        let mut stack = VecDeque::new();
        let mut values = HashSet::new();

        if let Some(node) = root {
            stack.push_front((Rc::clone(&node), 0));
        }

        while let Some((node, index)) = stack.pop_front() {
            values.insert(index);

            if let Some(left) = &node.borrow().left {
                stack.push_back((Rc::clone(left), index * 2 + 1));
            }
            if let Some(right) = &node.borrow().right {
                stack.push_back((Rc::clone(right), index * 2 + 2));
            }
        }
        FindElements { values }
    }

    fn find(&self, target: i32) -> bool {
        self.values.contains(&target)
    }
}

/**
 * Your FindElements object will be instantiated and called as such:
 * let obj = FindElements::new(root);
 * let ret_1: bool = obj.find(target);
 */

#[cfg(test)]
mod find_elements {
    use super::*;

    #[test]
    fn case_1() {
        let root = Some(Rc::new(RefCell::new(TreeNode {
            val: -1,
            left: None,
            right: Some(Rc::new(RefCell::new(TreeNode {
                val: -1,
                left: None,
                right: None,
            }))),
        })));

        let tree = FindElements::new(root);
        assert_eq!(tree.find(1), false);
        assert_eq!(tree.find(2), true);
    }
}

/**
 * https://leetcode.com/problems/recover-a-tree-from-preorder-traversal/
 */

// Definition for a binary tree node.
// #[derive(Debug, PartialEq, Eq)]
// pub struct TreeNode {
//     pub val: i32,
//     pub left: Option<Rc<RefCell<TreeNode>>>,
//     pub right: Option<Rc<RefCell<TreeNode>>>,
// }

// impl TreeNode {
//     #[inline]
//     pub fn new(val: i32) -> Self {
//         TreeNode {
//             val,
//             left: None,
//             right: None,
//         }
//     }
// }

impl Solution {
    pub fn recover_from_preorder(traversal: String) -> Option<Rc<RefCell<TreeNode>>> {
        let t: Vec<char> = traversal.chars().collect();
        let mut cursor = 0usize;
        let mut parents = vec![];
        let mut root = None;

        while cursor < t.len() {
            let mut i = cursor;

            while let Some(&v) = t.get(i) {
                if v != '-' {
                    break;
                }
                i += 1;
            }

            let depth = i - cursor;
            cursor = i;

            while let Some(&v) = t.get(i) {
                if v == '-' {
                    break;
                }
                i += 1;
            }

            let value: i32 = traversal[cursor..i].parse().unwrap();
            cursor = i;

            let node = Rc::new(RefCell::new(TreeNode::new(value)));

            let mut parent: Option<Rc<RefCell<TreeNode>>> = None;

            while let Some((p, d)) = parents.pop() {
                if depth > d {
                    parent = Some(p);
                    break;
                }
            }

            if let Some(p) = parent {
                parents.push((Rc::clone(&p), depth - 1));

                if p.borrow().left.is_none() {
                    p.borrow_mut().left = Some(Rc::clone(&node));
                } else {
                    p.borrow_mut().right = Some(Rc::clone(&node));
                }
            } else {
                root = Some(Rc::clone(&node));
            }

            parents.push((Rc::clone(&node), depth));
        }

        return root;
    }
}

#[cfg(test)]
mod recover_from_preorder {
    use super::*;

    #[test]
    fn case_1() {
        let result = Solution::recover_from_preorder(String::from("1-2-3"));
        let root = Rc::new(RefCell::new(TreeNode::new(1)));
        let left = Rc::new(RefCell::new(TreeNode::new(2)));
        let right = Rc::new(RefCell::new(TreeNode::new(3)));

        root.borrow_mut().left = Some(left);
        root.borrow_mut().right = Some(right);

        assert_eq!(result, Some(root));
    }
}

/**
 * https://leetcode.com/problems/construct-binary-tree-from-preorder-and-postorder-traversal/
 */
// use std::cell::RefCell;
// use std::rc::Rc;
/**
 *
 *
 */
// Definition for a binary tree node.
#[derive(Debug, PartialEq, Eq)]
pub struct TreeNode {
    pub val: i32,
    pub left: Option<Rc<RefCell<TreeNode>>>,
    pub right: Option<Rc<RefCell<TreeNode>>>,
}

impl TreeNode {
    #[inline]
    pub fn new(val: i32) -> Self {
        TreeNode {
            val,
            left: None,
            right: None,
        }
    }
}

impl Solution {
    pub fn construct_from_pre_post(
        preorder: Vec<i32>,
        postorder: Vec<i32>,
    ) -> Option<Rc<RefCell<TreeNode>>> {
        let mut parents: Vec<Rc<RefCell<TreeNode>>> = vec![];
        let mut root = None;
        let mut postorder = postorder.iter().peekable();

        for v in preorder {
            let node = Rc::new(RefCell::new(TreeNode::new(v)));

            if let Some(parent) = parents.pop() {
                if parent.borrow().left.is_none() {
                    parent.borrow_mut().left = Some(Rc::clone(&node));
                } else {
                    parent.borrow_mut().right = Some(Rc::clone(&node));
                }

                parents.push(Rc::clone(&parent));
            } else {
                root = Some(Rc::clone(&node));
            }

            parents.push(Rc::clone(&node));

            while let Some(&value) = postorder.peek() {
                if let Some(parent) = parents.pop() {
                    if parent.borrow().val == *value {
                        postorder.next();
                        continue;
                    } else {
                        parents.push(parent);
                        break;
                    }
                } else {
                    break;
                }
            }
        }

        return root;
    }
}

#[cfg(test)]
mod construct_from_pre_post {
    use super::*;

    #[test]
    fn case_0() {
        let result = Solution::construct_from_pre_post(vec![1, 2, 3], vec![2, 3, 1]);
        let root = Rc::new(RefCell::new(TreeNode::new(1)));
        let left = Rc::new(RefCell::new(TreeNode::new(2)));
        let right = Rc::new(RefCell::new(TreeNode::new(3)));

        root.borrow_mut().left = Some(left);
        root.borrow_mut().right = Some(right);

        assert_eq!(result, Some(root));
    }

    #[test]
    fn case_1() {
        let result = Solution::construct_from_pre_post(vec![1, 2, 4, 5], vec![4, 5, 2, 1]);
        let root = Rc::new(RefCell::new(TreeNode::new(1)));
        let left_0 = Rc::new(RefCell::new(TreeNode::new(2)));
        let left_1 = Rc::new(RefCell::new(TreeNode::new(4)));
        let right_1 = Rc::new(RefCell::new(TreeNode::new(5)));

        root.borrow_mut().left = Some(Rc::clone(&left_0));
        left_0.borrow_mut().left = Some(Rc::clone(&left_1));
        left_0.borrow_mut().right = Some(Rc::clone(&right_1));

        assert_eq!(result, Some(root));
    }

    #[test]
    fn case_2() {
        let result =
            Solution::construct_from_pre_post(vec![1, 2, 4, 5, 3, 6, 7], vec![4, 5, 2, 6, 7, 3, 1]);
        let root = Rc::new(RefCell::new(TreeNode::new(1)));
        let r_2 = Rc::new(RefCell::new(TreeNode::new(2)));
        let r_3 = Rc::new(RefCell::new(TreeNode::new(3)));
        let r_4 = Rc::new(RefCell::new(TreeNode::new(4)));
        let r_5 = Rc::new(RefCell::new(TreeNode::new(5)));
        let r_6 = Rc::new(RefCell::new(TreeNode::new(6)));
        let r_7 = Rc::new(RefCell::new(TreeNode::new(7)));

        root.borrow_mut().left = Some(Rc::clone(&r_2));
        root.borrow_mut().right = Some(Rc::clone(&r_3));
        r_2.borrow_mut().left = Some(Rc::clone(&r_4));
        r_2.borrow_mut().right = Some(Rc::clone(&r_5));
        r_3.borrow_mut().left = Some(Rc::clone(&r_6));
        r_3.borrow_mut().right = Some(Rc::clone(&r_7));

        assert_eq!(result, Some(root));
    }
}

/**
 * https://leetcode.com/problems/most-profitable-path-in-a-tree/
 */
// use std::collections::{HashMap, HashSet, VecDeque};
// use std::rc::Rc;

enum List {
    Cons(i32, Rc<List>),
    Nil,
}

impl Solution {
    pub fn most_profitable_path(edges: Vec<Vec<i32>>, bob: i32, amounts: Vec<i32>) -> i32 {
        // build the tree

        let mut tree = HashMap::new();

        for edge in edges {
            let (u, v) = (edge[0], edge[1]);
            tree.entry(u).or_insert(Vec::new()).push(v);
            tree.entry(v).or_insert(Vec::new()).push(u);
        }

        // bfs find way "bob" -> 0

        let mut bob_visited = HashSet::new();
        let mut bob_to_visit = VecDeque::new();
        let mut probably_bob_path = None;

        bob_to_visit.push_front((vec![bob], Rc::new(List::Nil)));

        while let Some((nodes, way)) = bob_to_visit.pop_front() {
            for node in nodes {
                let new_way = Rc::new(List::Cons(node, way.clone()));

                if node == 0 {
                    probably_bob_path = Some(Rc::clone(&new_way));
                    break;
                }

                // tree[node] has to be defined - by construction
                let adjacent = tree.get(&node).unwrap();

                bob_visited.insert(node);

                let new_nodes = adjacent
                    .iter()
                    .filter(|n| !bob_visited.contains(n))
                    .map(|n| *n)
                    .collect();

                bob_to_visit.push_back((new_nodes, new_way));
            }

            if probably_bob_path.is_some() {
                break;
            }
        }

        let mut bob_path: VecDeque<i32> = VecDeque::new();

        if let Some(mut root) = probably_bob_path {
            while let List::Cons(node, next) = root.as_ref() {
                bob_path.push_front(*node);
                root = next.clone();
            }
        } else {
            panic!("bob cannot get to 0")
        }

        let bob_time = bob_path
            .iter()
            .enumerate()
            .map(|(index, node)| (*node, index))
            .collect::<HashMap<_, _>>();

        // bfs find most profitable alice path

        let mut alice_visited = HashSet::new();
        let mut alice_to_visit = VecDeque::new();
        alice_to_visit.push_front((vec![0], (0, 0)));
        let mut max: Option<i32> = None;

        while let Some((nodes, (step, step_amount))) = alice_to_visit.pop_front() {
            for node in nodes {
                // tree[node] has to be defined - by construction
                let adjacent = tree.get(&node).unwrap();
                let amount = amounts[node as usize];
                alice_visited.insert(node);
                let mut new_amount = step_amount;

                if let Some(&bob_at) = bob_time.get(&node) {
                    if bob_at == step {
                        new_amount += amount / 2;
                    } else if bob_at > step {
                        new_amount += amount;
                    }
                } else {
                    new_amount += amount;
                }

                let new_nodes: Vec<i32> = adjacent
                    .iter()
                    .filter(|n| !alice_visited.contains(n))
                    .map(|n| *n)
                    .collect();

                if new_nodes.len() == 0 {
                    if let Some(m) = max {
                        max = Some(m.max(new_amount));
                    } else {
                        max = Some(new_amount);
                    }
                }

                alice_to_visit.push_back((new_nodes, (step + 1, new_amount)));
            }
        }

        if let Some(max) = max {
            return max;
        } else {
            panic!("there is no solution");
        }
    }
}

#[cfg(test)]
mod most_profitable_path {
    use super::*;

    #[test]
    fn case_1() {
        let result = Solution::most_profitable_path(
            vec![vec![0, 1], vec![1, 2], vec![1, 3], vec![3, 4]],
            3,
            vec![-2, 4, 2, -4, 6],
        );
        assert_eq!(result, 6);
    }

    #[test]
    fn case_2() {
        let result = Solution::most_profitable_path(vec![vec![0, 1]], 1, vec![-1000, 2000]);
        assert_eq!(result, -1000);
    }

    #[test]
    fn case_3() {
        let result =
            Solution::most_profitable_path(vec![vec![0, 1], vec![1, 2]], 2, vec![10, 20, 10]);
        assert_eq!(result, 20);
    }
}

/**
 * https://leetcode.com/problems/number-of-sub-arrays-with-odd-sum/
 */

/*

1 2 3 4

subarray counts:
1 - 1
2 - 3
3 - 6
4 - 10
5 - 15
6 - 21
7 - 28
8 - 36
n (n+1) /2


1 - (odd: 1)
1 2 - (odd: 1) + (odd: 1, even: 1) = (odd: 2, even: 1)
1 2 3 - (odd: 2, even: 1) + (odd: 2, even: 1) = (odd: 4, even: 2)
1 2 3 4 - (odd: 4, even: 2) + (odd: 2, even: 2) = (odd: 6, even: 2)
1 2 3 4 5 - (odd: 6, even: 4) + (odd: 3, even: 2) = (odd: 9, even: 6)

1 - 1 - (1, 0)
1 1 - 2 - (1, 0) + (1, 1) = (2, 1)
1 1 1 - 3 - (2, 1) + (2, 1) = (4, 2)

1 1 1 1 - 4 - (4, 2) + (2, 2) = (6, 4)
1 1 1 2 - 5 - (4, 2) + (2, 2) = (6, 4)

1 1 1 1 1 - 5 - (6, 4) + (3, 2) = (9, 6)
1 1 1 1 2 - 6 - (6, 4) + (2, 3) = (9, 6)

1 1 1 1 1 1 - 6 - (9, 6) + (3, 3) = (12, 9)
1 1 1 1 1 2 - 7 - (9, 6) + (3, 3) = (12, 9)

1 1 1 1 2 2 - 8 - (9, 6) + (2, 4) = (11, 10)
1 1 1 1 2 2 2 - 10 - (11, 10) + (2, 5) = (13, 15)
1 1 1 1 2 2 2 1 - 11 - (13, 15) + (6, 2) = (19, 17)

1 1 1 1 1 1 1 - 7 - (12, 9) + (4, 3) = (16, 12)
1 1 1 1 1 1 2 - 8 - (12, 9) + (3, 4) = (15, 13)


so the algorithm is:
- for each item of array
    - if item is even
        - increase even counter
    - if item is odd
        - swap even and odd counter
        - increase odd counter
    - accumulate odd sum with counter

 */
impl Solution {
    pub fn num_of_subarrays(arr: Vec<i32>) -> i32 {
        let mut odd = 0;
        let mut even = 0;
        let mut sum: i32 = 0;

        for n in arr {
            if n % 2 == 0 {
                even += 1;
            } else {
                (even, odd) = (odd, even);
                odd += 1;
            }

            sum = (sum + odd) % 1_000_000_007;
        }

        return sum;
    }
}

#[cfg(test)]
mod num_of_subarrays {
    use super::*;

    #[test]
    fn case_1() {
        let result = Solution::num_of_subarrays(vec![1, 3, 5]);
        assert_eq!(result, 4);
    }

    #[test]
    fn case_2() {
        let result = Solution::num_of_subarrays(vec![2, 4, 6]);
        assert_eq!(result, 0);
    }

    #[test]
    fn case_3() {
        let result = Solution::num_of_subarrays(vec![1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(result, 16);
    }
}

/**
 * https://leetcode.com/problems/maximum-absolute-sum-of-any-subarray/
 */
impl Solution {
    pub fn max_absolute_sum(nums: Vec<i32>) -> i32 {
        let (mut cmax, mut max) = (0, 0);
        let (mut cmin, mut min) = (0, 0);

        for n in nums {
            cmax = n.max(cmax + n);
            max = max.max(cmax);
            cmin = n.min(cmin + n);
            min = min.min(cmin);
        }

        return max.max(-min);
    }
}

#[cfg(test)]
mod max_absolute_sum {
    use super::*;

    #[test]
    fn case_1() {
        let result = Solution::max_absolute_sum(vec![1, -3, 2, 3, -4]);
        assert_eq!(result, 5);
    }

    #[test]
    fn case_2() {
        let result = Solution::max_absolute_sum(vec![2, -5, 1, -4, 3, -2]);
        assert_eq!(result, 8);
    }

    #[test]
    fn case_3() {
        let result = Solution::max_absolute_sum(vec![1, -3, 4, -2, -1, 6]);
        assert_eq!(result, 9);
    }
}

/**
 * https://leetcode.com/problems/letter-tile-possibilities/
 */

/*

AAB - 8
A B
AA AB BA
AAB ABA BAA

AAA - 3
A
AA
AAA

ABC - 15
A B C
AB AC BC BA CA CB
ABC ACB BAC BCA CAB CBA

*/
impl Solution {
    pub fn num_tile_possibilities(tiles: String) -> i32 {
        let tiles: Vec<char> = tiles.chars().collect();
        let mut unique = HashSet::new();
        let frequencies = tiles.iter().fold(HashMap::new(), |mut map, value| {
            map.entry(value).and_modify(|f| *f += 1).or_insert(1);
            map
        });

        fn backtrack(
            frequencies: &HashMap<&char, i32>,
            unique: &mut HashSet<String>,
            letters: &Vec<char>,
            current: &mut Vec<char>,
            n: usize,
            i: usize,
        ) -> bool {
            if i == n {
                return true;
            }

            for letter in letters {
                let freq = frequencies.get(letter).unwrap();
                let count = current
                    .iter()
                    .fold(0, |a, v| if v == letter { a + 1 } else { a });

                if count >= *freq {
                    continue;
                }

                current.push(*letter);
                let the_end = backtrack(frequencies, unique, letters, current, n, i + 1);
                if the_end {
                    unique.insert(current.iter().collect());
                }

                current.pop();
            }

            return false;
        }

        for i in 1..=tiles.len() {
            backtrack(&frequencies, &mut unique, &tiles, &mut vec![], i, 0);
        }

        return unique.len() as i32;
    }
}

#[cfg(test)]
mod num_tile_possibilities {
    use super::*;

    #[test]
    fn case_1() {
        let result = Solution::num_tile_possibilities(String::from("AAB"));
        assert_eq!(result, 8);
    }

    #[test]
    fn case_2() {
        let result = Solution::num_tile_possibilities(String::from("AAABBC"));
        assert_eq!(result, 188);
    }

    #[test]
    fn case_3() {
        let result = Solution::num_tile_possibilities(String::from("V"));
        assert_eq!(result, 1);
    }
}

/**
 * https://leetcode.com/problems/length-of-longest-fibonacci-subsequence/
 */

/*

// 1 2 3 5 6 8 13

1

1 - (None, 1)

1 2

2 - (None, 1)
3 - (2, 2)

1 2 3

5 - (3, 3) // can be omitted
4 - (3, 2)

1 2 3 5

5 - (None, 1)
8 - (5, 4)
8 - (5, 3) // redundant
7 - (5, 2)
6 - (5, 2) // can be omitted

1 2 3 5 6

11 - (6, 3)
11 - (6, 2) // redundant
9 - (6, 2)
8 - [(5, 3), (6, 2)] // collision - use longest
7 - [(5, 2), (6, 2)] // collision - save both

1 2 3 5 6 7

12 - [(7, 3)]
13 - [(7, 3)]

7 - [(5, 2), (6, 2)] // collision - save both



naive approach:

- max_len
- iterate (i) over each element O(n)
    - look for [arr[i]] in hashmap O(1)
        - if there are some pairs with suitable sum
            - calculate new sums of [arr[i] + last_in_pair] = (arr[i], new_len) O(1)
            - max_len = max_len.max(new_len) O(1)
    - iterate (j) from 0 to (i - 1) O(n)
        - if !hashmap.contains([arr[i] + arr[j]])
            - add [arr[i] + arr[j]] = (arr[j], 2) to hashmap - as started pair

O(n^2)


*/
impl Solution {
    pub fn len_longest_fib_subseq_top(arr: Vec<i32>) -> i32 {
        let mut dp = HashMap::new();
        let set: HashSet<i32> = arr.iter().cloned().collect();
        let mut max_len = 0;

        for i in 0..arr.len() {
            for j in 0..i {
                let x = arr[j];
                let y = arr[i];
                let z = y - x;

                if z < x && set.contains(&z) {
                    let len = *dp.get(&(z, x)).unwrap_or(&2) + 1;
                    dp.insert((x, y), len);
                    max_len = max_len.max(len);
                }
            }
        }

        max_len
    }

    pub fn len_longest_fib_subseq(arr: Vec<i32>) -> i32 {
        let mut numbers: HashSet<i32> = arr.iter().cloned().collect();

        for &n in &arr {
            numbers.insert(n);
        }

        fn search(a: i32, b: i32, length: i32, numbers: &HashSet<i32>) -> i32 {
            if numbers.contains(&(a + b)) {
                return search(b, a + b, length + 1, numbers);
            }
            return length;
        }

        let mut max_length = 0;

        for i in 0..arr.len() - 1 {
            for j in i + 1..arr.len() {
                let length = search(arr[i], arr[j], 2, &numbers);
                max_length = max_length.max(length);
            }
        }

        if max_length > 2 {
            max_length
        } else {
            0
        }
    }

    pub fn len_longest_fib_subseq_takes_to_long(arr: Vec<i32>) -> i32 {
        let mut sums: HashMap<i32, HashMap<i32, i32>> = HashMap::new();
        let mut max_len = 0;

        for (i, &n) in arr.iter().enumerate() {
            let mut new_chains: HashMap<i32, HashMap<i32, i32>> = HashMap::new();

            if let Some(chains) = sums.get(&n) {
                for (last, length) in chains {
                    let mut new_chain = HashMap::new();
                    new_chain.insert(n, length + 1);
                    new_chains.insert(last + n, new_chain);
                    max_len = max_len.max(length + 1);
                }
            }

            for (k, v) in new_chains {
                let chains = sums.entry(k).or_insert(HashMap::new());
                for (s, c) in v {
                    chains.insert(s, c);
                }
            }

            for j in 0..i {
                let sum = arr[j] + n;
                if let Some(chains) = sums.get(&sum) {
                    if chains.contains_key(&n) {
                        continue;
                    }
                }
                let pairs = sums.entry(sum).or_insert(HashMap::new());
                pairs.insert(n, 2);
            }
        }

        return max_len;
    }
}

#[cfg(test)]
mod len_longest_fib_subseq {
    use super::*;

    #[test]
    fn case_1() {
        let result = Solution::len_longest_fib_subseq(vec![1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(result, 5);
    }

    #[test]
    fn case_2() {
        let result = Solution::len_longest_fib_subseq(vec![1, 3, 7, 11, 12, 14, 18]);
        assert_eq!(result, 3);
    }

    #[test]
    fn case_3() {
        let result =
            Solution::len_longest_fib_subseq(vec![2, 4, 7, 8, 9, 10, 14, 15, 18, 23, 32, 50]);
        assert_eq!(result, 5);
    }

    #[test]
    fn case_4() {
        let result = Solution::len_longest_fib_subseq(vec![
            193, 3931, 5403, 6322, 7997, 11671, 12075, 12744, 14929, 19780, 19892, 22630, 22891,
            23108, 24076, 24737, 25008, 25524, 25571, 25927, 26366, 26488, 27456, 28770, 29243,
            29364, 31211, 34344, 34395, 35438, 35782, 36500, 36736, 38235, 38495, 39058, 39428,
            39741, 40933, 42510, 42647, 43080, 43511, 45049, 45690, 46228, 48837, 48917, 49214,
            51299, 51564, 51884, 56146, 59084, 61175, 61970, 64549, 64580, 64604, 67854, 68259,
            72513, 72519, 72553, 74768, 75338, 81358, 81752, 82646, 83043, 84231, 85860, 89288,
            90639, 91166, 92511, 94656, 99092, 99995, 100641, 100651, 100883, 102871, 105682,
            106356, 106402, 107960, 109163, 109561, 110146, 111614, 114555, 115521, 115962, 117211,
            118344, 120974, 124171, 125252, 126891, 128424, 128564, 129509, 130150, 130472, 130846,
            131004, 131965, 132018, 132029, 133675, 137205, 137457, 139003, 139801, 143190, 143415,
            143939, 144133, 145500, 146562, 148510, 151133, 152096, 153737, 153950, 161192, 161682,
            163066, 163684, 166017, 168298, 170078, 172226, 172459, 172480, 174367, 175769, 179551,
            179848, 179890, 180621, 183143, 183451, 184220, 184654, 186526, 187215, 192908, 192915,
            193011, 197285, 197861, 198818, 199279, 199562, 202490, 204717, 205441, 208252, 208266,
            209076, 209124, 212614, 213017, 213626, 215479, 215534, 216435, 217411, 219537, 220416,
            221314, 223369, 223589, 223607, 226185, 227154, 228939, 229355, 230865, 234005, 235688,
            236374, 239000, 239048, 239121, 240533, 240594, 240857, 241017, 242479, 243428, 245782,
            248591, 249049, 249589, 251441, 252343, 253349, 257086, 257973, 258106, 258215, 258728,
            262357, 262853, 263712, 266381, 267986, 269300, 270847, 271135, 272491, 274534, 275071,
            275755, 276040, 278542, 279901, 282508, 283594, 283841, 284062, 285110, 288409, 289951,
            290495, 291005, 293485, 293714, 296041, 296998, 298301, 298547, 298707, 302452, 303668,
            304790, 305085, 305760, 306250, 308184, 309427, 310983, 312492, 315395, 315503, 318208,
            319058, 320243, 322594, 323234, 325609, 325965, 326113, 326661, 330505, 331230, 331429,
            332978, 333538, 334240, 337022, 337891, 341330, 342008, 342146, 343983, 344973, 345292,
            345657, 346504, 346561, 347477, 348605, 348956, 349051, 350624, 350878, 351424, 351485,
            352080, 353307, 353426, 353607, 357398, 361221, 361345, 361873, 363861, 365475, 366154,
            366295, 370065, 370702, 371322, 371444, 373826, 374644, 380984, 381646, 381887, 381945,
            381983, 383219, 384169, 384968, 385012, 385174, 385807, 386344, 387186, 388121, 389613,
            390689, 391064, 391689, 392124, 393943, 395060, 397418, 398184, 400038, 400956, 401511,
            402690, 402941, 404015, 404181, 404760, 413584, 415003, 415499, 416069, 417094, 422957,
            423877, 424938, 425647, 426276, 427871, 428195, 428328, 428996, 429057, 429988, 430071,
            431325, 432716, 433637, 438553, 440740, 441348, 442130, 444521, 445693, 448441, 448816,
            449665, 450230, 450668, 450859, 451369, 451505, 453591, 455938, 458659, 460375, 463112,
            463730, 464288, 465679, 466666, 467144, 468405, 468731, 478393, 478478, 479697, 480206,
            480449, 480555, 480975, 481896, 482424, 483519, 483826, 485611, 485906, 486413, 486757,
            487005, 488291, 490062, 491717, 493808, 494485, 495561, 496246, 498443, 502074, 504638,
            504856, 505570, 506150, 506559, 507897, 510584, 515153, 516724, 517102, 517221, 517626,
            518351, 519001, 519372, 519897, 520223, 520547, 521934, 523547, 523751, 526338, 527185,
            528991, 529874, 530407, 530465, 530920, 533417, 533438, 534110, 534746, 537532, 539540,
            540145, 540650, 541015, 541027, 543009, 546842, 548124, 548982, 552825, 554980, 557331,
            557502, 558853, 560351, 564300, 565099, 568203, 568612, 570336, 571641, 572928, 573623,
            573749, 574166, 574772, 575801, 575848, 576098, 576278, 576413, 576542, 577138, 577410,
            579796, 579897, 580539, 580821, 581101, 581205, 581348, 582134, 582146, 582875, 583923,
            584132, 584169, 586196, 591172, 591750, 593063, 595255, 595838, 596029, 596052, 596122,
            597558, 599693, 600612, 600686, 601585, 602092, 604586, 606244, 606537, 608302, 610142,
            614013, 614941, 615020, 615098, 616995, 619305, 619777, 620381, 620804, 622245, 622661,
            623094, 623986, 626259, 626299, 626893, 627720, 628863, 630175, 633425, 634941, 636237,
            639146, 639646, 641192, 641612, 641733, 642642, 643522, 643732, 644709, 644948, 645691,
            645775, 647216, 647364, 647591, 647616, 651072, 655002, 656934, 657999, 658130, 660661,
            662684, 663094, 663700, 666305, 669039, 669127, 673693, 675009, 675983, 677228, 677275,
            678454, 678831, 679455, 681779, 681814, 683663, 686061, 686321, 689773, 694308, 694836,
            694961, 697024, 699141, 699356, 699674, 699806, 701684, 702098, 703135, 704420, 706762,
            707372, 708648, 709688, 710720, 711180, 711566, 713892, 714972, 718685, 719366, 719373,
            724525, 724829, 725190, 725699, 725825, 726643, 728624, 728851, 731291, 734850, 735485,
            736484, 736927, 737222, 738323, 738545, 739470, 740625, 740898, 746583, 747473, 750673,
            752783, 753596, 754923, 755699, 757000, 760484, 760664, 761580, 764466, 765283, 765575,
            767256, 768841, 770759, 770766, 771293, 771910, 774473, 774631, 775195, 777333, 778292,
            778584, 778651, 778999, 780101, 781410, 781822, 783206, 784446, 787142, 787940, 788066,
            790294, 791143, 792441, 792450, 795950, 796484, 797203, 797675, 798214, 799065, 801012,
            801232, 801253, 802510, 802580, 802922, 804754, 805224, 805438, 807084, 809421, 811290,
            813701, 813937, 813952, 815540, 815730, 819294, 820565, 821550, 823523, 825489, 827280,
            828038, 828141, 831686, 831973, 833100, 833308, 833743, 834701, 834845, 835058, 835173,
            835948, 837198, 837996, 839484, 839961, 844423, 844861, 849669, 850382, 852277, 852473,
            853075, 854744, 855756, 856294, 856357, 857892, 860043, 860993, 861740, 862661, 865736,
            866371, 867373, 867835, 869086, 872230, 873360, 874664, 875945, 882375, 883892, 884144,
            887296, 887689, 889738, 890756, 892243, 893755, 894713, 895803, 895874, 896497, 898649,
            899706, 900309, 901338, 901832, 904221, 904331, 904595, 908126, 908882, 909520, 912231,
            912410, 913488, 915524, 916243, 916525, 918731, 920328, 920549, 922821, 922905, 924210,
            927313, 927693, 928443, 930682, 930940, 933091, 933804, 935342, 937635, 943838, 946596,
            948084, 950776, 952687, 957120, 959030, 959104, 959806, 960320, 960888, 965320, 966881,
            967226, 969456, 970217, 972541, 972669, 973081, 974470, 974946, 975376, 976032, 976996,
            978372, 979581, 979893, 982781, 984130, 985018, 985073, 985134, 988381, 989187, 992251,
            994086, 995235, 997916, 998948,
        ]);
        assert_eq!(result, 5);
    }
}

/**
 * https://leetcode.com/problems/shortest-common-supersequence/
 */

/*


abac, cab -> cabac
aba, b -> aba
a, bab -> bab
abc, bcd -> abcd

let we have two cursors for positions in strings respectively

either
1. move cursor in the first string - push char to result
    if a char in the second string cursor position is equal
        move cursor in the second string
    else
        move cursor in the second string to beginning

    starting from cursor position if the second string - push char to result

2. move cursor in the second string
    ... same but for the first string

3. return the shortest result string


*/
impl Solution {
    pub fn shortest_common_supersequence(str1: String, str2: String) -> String {
        let str1 = str1.as_bytes();
        let str2 = str2.as_bytes();

        /*

          c a b
        a 0 1 1
        b 0 1 2
        a 0 1 2
        c 1 1 2

        c a b a c

          a b c
        d 0 0 0
        e 0 0 0
        f 0 0 0

        a d b e c f

          a b c
        d 0 0 0
        b 0 1 1
        c 0 1 2

        a d b c

          a b c
        d 0 0 0
        b 0 1 1
        f 0 1 1

        a d b c f

          a b c
        a 1 1 1
        b 1 2 2


        */

        let mut lcs = vec![vec![0; str2.len()]; str1.len()];

        for (i, &c1) in str1.iter().enumerate() {
            for (j, &c2) in str2.iter().enumerate() {
                if c1 == c2 {
                    if i > 0 && j > 0 {
                        lcs[i][j] = lcs[i - 1][j - 1] + 1;
                    } else {
                        lcs[i][j] = 1;
                    }
                } else {
                    if i > 0 && j > 0 {
                        lcs[i][j] = lcs[i - 1][j].max(lcs[i][j - 1]);
                    } else if i > 0 {
                        lcs[i][j] = lcs[i - 1][j];
                    } else if j > 0 {
                        lcs[i][j] = lcs[i][j - 1];
                    } else {
                        lcs[i][j] = 0;
                    }
                }
            }
        }

        let (mut i, mut j) = (str1.len() - 1, str2.len() - 1);
        let mut result = vec![];
        let mut last_s1_poped = false;
        let mut last_s2_poped = false;

        while i > 0 || j > 0 {
            if i > 0 && j > 0 {
                if lcs[i - 1][j] == lcs[i][j - 1] {
                    if lcs[i][j] != lcs[i - 1][j] {
                        j -= 1;
                    }
                    result.push(str1[i]);
                    i -= 1;
                    continue;
                }
                if lcs[i - 1][j] > lcs[i][j - 1] {
                    result.push(str1[i]);
                    i -= 1;
                    continue;
                }
                if lcs[i - 1][j] < lcs[i][j - 1] {
                    result.push(str2[j]);
                    j -= 1;
                    continue;
                }
            }
            if i > 0 {
                if i > 0 && lcs[i - 1][j] < lcs[i][j] {
                    last_s2_poped = true;
                }
                result.push(str1[i]);
                i -= 1;
            }
            if j > 0 {
                if j > 0 && lcs[i][j - 1] < lcs[i][j] {
                    last_s1_poped = true;
                }
                result.push(str2[j]);
                j -= 1;
            }
        }

        if lcs[0][0] == 1 {
            result.push(str1[0]);
        } else {
            if !last_s1_poped {
                result.push(str1[0]);
            }

            if !last_s2_poped {
                result.push(str2[0]);
            }
        }

        let result: Vec<u8> = result.iter().copied().rev().collect();
        return String::from(str::from_utf8(&result).unwrap());
    }
}

#[cfg(test)]
mod shortest_common_supersequence {
    use super::*;

    #[test]
    fn case_1() {
        let result =
            Solution::shortest_common_supersequence(String::from("abac"), String::from("cab"));
        assert_eq!(result, String::from("cabac"));
    }

    #[test]
    fn case_2() {
        let result =
            Solution::shortest_common_supersequence(String::from("aba"), String::from("b"));
        assert_eq!(result, String::from("aba"));
    }

    #[test]
    fn case_3() {
        let result = Solution::shortest_common_supersequence(
            String::from("aaaaaaaa"),
            String::from("aaaaaaaa"),
        );
        assert_eq!(result, String::from("aaaaaaaa"));
    }

    #[test]
    fn case_4() {
        let result = Solution::shortest_common_supersequence(
            String::from("bbbaaaba"),
            String::from("bbababbb"),
        );
        assert_eq!(result, String::from("bbbaaababbb"));
    }

    #[test]
    fn case_5() {
        let result = Solution::shortest_common_supersequence(
            String::from("bcacaaab"),
            String::from("bbabaccc"),
        );
        assert_eq!(result, String::from("bcacaababaccc"));
    }

    #[test]
    fn case_6() {
        let result = Solution::shortest_common_supersequence(
            String::from("accabcba"),
            String::from("aacbbbbbaa"),
        );
        assert_eq!(result, String::from("aaccabbbbcbaa"));
    }
}

/**
 * https://leetcode.com/problems/longest-common-subsequence/
 */
impl Solution {
    pub fn longest_common_subsequence(text1: String, text2: String) -> i32 {
        let str1 = text1.as_bytes();
        let str2 = text2.as_bytes();

        /*

          c a b
        a 0 1 1
        b 0 1 2
        a 0 1 2
        c 1 1 2

        */

        let mut lcs = vec![vec![0; str2.len()]; str1.len()];

        for (i, &c1) in str1.iter().enumerate() {
            for (j, &c2) in str2.iter().enumerate() {
                if c1 == c2 {
                    if i > 0 && j > 0 {
                        lcs[i][j] = lcs[i - 1][j - 1] + 1;
                    } else {
                        lcs[i][j] = 1;
                    }
                } else {
                    if i > 0 && j > 0 {
                        lcs[i][j] = lcs[i - 1][j].max(lcs[i][j - 1]);
                    } else if i > 0 {
                        lcs[i][j] = lcs[i - 1][j];
                    } else if j > 0 {
                        lcs[i][j] = lcs[i][j - 1];
                    } else {
                        lcs[i][j] = 0;
                    }
                }
            }
        }

        lcs[str1.len()][str2.len()]
    }
}
