use std::cmp::{self, Ordering};

pub struct Solution {}

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
use std::collections::{BTreeMap, BTreeSet, BinaryHeap, HashMap};

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
// use std::collections::BinaryHeap;
use std::cmp::Reverse;

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
    fn find_place(left: &mut BinaryHeap<i32>, s: &mut Vec<i32>) -> () {
        let first_empty = s
            .iter()
            .enumerate()
            .find(|(_, &i)| i == 0)
            .map(|(index, _)| index);
    }

    pub fn construct_distanced_sequence(n: i32) -> Vec<i32> {
        let n: usize = n.try_into().unwrap();
        let sequence = vec![0u32; n * 2 + 1];

        return vec![];
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
}
