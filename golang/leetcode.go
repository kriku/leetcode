package leetcode

import (
	"math"
	"slices"
)

// https://leetcode.com/problems/apply-operations-to-an-array/
/*
2 <= nums.length <= 2000
0 <= nums[i] <= 1000
*/
func applyOperations(nums []int) []int {
	// first operation - merge pairs
	for i := 1; i < len(nums); i++ {
		if nums[i-1] == nums[i] {
			nums[i-1] = nums[i-1] + nums[i]
			nums[i] = 0
		}
	}

	// second operation - move all zeros to the end
	for i, z := 0, 0; i < len(nums); i++ {
		if nums[i] == 0 {
			z += 1
		}

		if z != 0 && nums[i] != 0 {
			nums[i-z] = nums[i]
			nums[i] = 0
		}
	}

	return nums
}

// https://leetcode.com/problems/merge-two-2d-arrays-by-summing-values/description/?envType=daily-question&envId=2025-03-02
/*
1 <= nums1.length, nums2.length <= 200
nums1[i].length == nums2[j].length == 2
1 <= id_i, val_i <= 1000
*/
func mergeArrays(nums1 [][]int, nums2 [][]int) [][]int {
	result := make([][]int, 0)

	for p1, p2 := 0, 0; p1 < len(nums1) || p2 < len(nums2); {
		if p1 < len(nums1) && p2 < len(nums2) {
			if nums1[p1][0] < nums2[p2][0] {
				result = append(result, nums1[p1])
				p1++
				continue
			}
			if nums1[p1][0] > nums2[p2][0] {
				result = append(result, nums2[p2])
				p2++
				continue
			}
			if nums1[p1][0] == nums2[p2][0] {
				result = append(result, []int{nums1[p1][0], nums1[p1][1] + nums2[p2][1]})
				p1++
				p2++
				continue
			}
		}

		if p2 == len(nums2) {
			result = append(result, nums1[p1])
			p1++
			continue
		}

		if p1 == len(nums1) {
			result = append(result, nums2[p2])
			p2++
			continue
		}
	}

	return result
}

// https://leetcode.com/problems/partition-array-according-to-given-pivot/
func pivotArray(nums []int, pivot int) []int {
	result := make([]int, 0, len(nums))

	for _, n := range nums {
		if n < pivot {
			result = append(result, n)
		}
	}

	for _, n := range nums {
		if n == pivot {
			result = append(result, n)
		}
	}

	for _, n := range nums {
		if n > pivot {
			result = append(result, n)
		}
	}

	return result
}

// https://leetcode.com/problems/check-if-number-is-a-sum-of-powers-of-three/
func checkPowerOfThree(sum, i, n int) bool {
	if sum == n {
		return true
	}
	if sum > n || i > n {
		return false
	}
	return checkPowerOfThree(sum+i, i*3, n) || checkPowerOfThree(sum, i*3, n)
}

/*
1 <= n <= 10^7
*/

func checkPowersOfThree(n int) bool {
	return checkPowerOfThree(0, 1, n)
}

// https://leetcode.com/problems/count-total-number-of-colored-cells/
/*
1 => 1
2 => 5 (+4)
3 => 13 (+8)
4 => 25 (+12)
5 => 41 (+16)
...
n => 1 + 4 + 8 + 12 + ... + 4*(n-1)
n => 1 + 4 * (1 + 2 + 3 + ... + (n-1))
n => 1 + 4 * (n - 1) * n/2
n => 1 + 2 * (n - 1) * n
*/
/*
1 <= n <= 10^5
*/
func coloredCells(n int) int64 {
	return int64(1 + 2*(n-1)*n)
}

// https://leetcode.com/problems/sqrtx/
func mySqrt(x int) int {
	left, right := 1, x

	for {
		mid := left + (right-left)/2

		if mid*mid > x {
			right = mid
		} else {
			if (mid+1)*(mid+1) > x {
				return mid
			}
			left = mid + 1
		}
	}
}

// https://leetcode.com/problems/find-missing-and-repeated-values/
func findMissingAndRepeatedValues(grid [][]int) []int {
	numbers := make([]bool, len(grid)*len(grid[0]))
	result := make([]int, 0)

	for _, row := range grid {
		for _, v := range row {
			if numbers[v-1] {
				result = append(result, v)
			}
			numbers[v-1] = true
		}
	}

	for i := range numbers {
		if !numbers[i] {
			result = append(result, i+1)
		}
	}

	return result
}

func isPrime(n int) bool {
	if n == 1 {
		return false
	}

	for i := 2; i <= int(math.Round(math.Sqrt(float64(n)))); i++ {
		if n%i == 0 {
			return false
		}
	}
	return true
}

// https://leetcode.com/problems/closest-prime-numbers-in-range/
/*
1 <= left <= right <= 10^6
*/
func closestPrimes(left int, right int) []int {
	last, min := 0, right-left+1
	result := make([]int, 2)
	result[0], result[1] = -1, -1

	for i := left; i <= right; i++ {
		if isPrime(i) {
			if last > 0 {
				if min > i-last {
					result[0] = last
					result[1] = i
					min = i - last
				}
				if min <= 2 {
					return result
				}
			}
			last = i
		}
	}

	return result
}

// https://leetcode.com/problems/minimum-recolors-to-get-k-consecutive-black-blocks/
func minimumRecolors(blocks string, k int) int {
	w, min := 0, k

	for i, c := range blocks {
		if c == 'W' {
			w++
		}

		if i >= k {
			if blocks[i-k] == 'W' {
				w--
			}
		}

		if i+1 >= k {
			if min > w {
				min = w
			}
		}
	}

	return min
}

// https://leetcode.com/problems/alternating-groups-ii/
/*
3 <= colors.length <= 10^5
0 <= colors[i] <= 1
3 <= k <= colors.length
*/
func numberOfAlternatingGroups(colors []int, k int) int {
	n := len(colors)

	alt, groups, prev := 0, 0, -1

	for i := range n + k - 1 {
		index := i % n

		if prev != colors[index] {
			alt += 1
		} else {
			alt = 1
		}

		if alt >= k {
			groups += 1
		}

		prev = colors[index]
	}

	return groups
}

// https://leetcode.com/problems/count-of-substrings-containing-every-vowel-and-k-consonants-ii/
func isAllVowels(v1, v2 []int) bool {
	isAll := true

	for a := range 5 {
		if v1[a] == v2[a] {
			isAll = false
		}
	}

	return isAll
}

func countOfSubstrings(word string, k int) int64 {
	n := len(word)
	consonants := 0
	cons := make([]int, n+1)
	next := make([]int, n+1)
	cons[0] = 0

	as, es, is, os, us := 0, 0, 0, 0, 0

	v := make([][]int, n+1)
	v[0] = []int{0, 0, 0, 0, 0}

	for i, c := range word {
		switch c {
		case 'a':
			as += 1
		case 'e':
			es += 1
		case 'i':
			is += 1
		case 'o':
			os += 1
		case 'u':
			us += 1
		default:
			consonants += 1
		}
		cons[i+1] = consonants
		v[i+1] = []int{as, es, is, os, us}
	}

	for i, last := n, n+1; i > 0; i-- {
		next[i] = last
		c := word[i-1]
		if c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u' {
			continue
		}
		last = i
	}
	next[0] = next[1]

	result := int64(0)

	i, j := 0, 0
	for j < n+1 {
		if isAllVowels(v[i], v[j]) && cons[j]-cons[i] == k {
			result += int64(next[j] - j)
		}

		// shrink window
		if isAllVowels(v[i], v[j]) && cons[j]-cons[i] == k {
			i += 1
			continue
		}
		if cons[j]-cons[i] > k {
			i += 1
			continue
		}

		// expand window
		j += 1
	}

	return result
}

func countABC(r rune, a, b, c *int, delta int) {
	switch r {
	case 'a':
		*a += delta
	case 'b':
		*b += delta
	case 'c':
		*c += delta
	}
}

// https://leetcode.com/problems/number-of-substrings-containing-all-three-characters/
func numberOfSubstrings(s string) int {
	n := 0

	a, b, c := 0, 0, 0

	left := 0

	for i, r := range s {
		countABC(r, &a, &b, &c, 1)

		for a > 0 && b > 0 && c > 0 && left < len(s) {
			n += len(s) - i
			countABC(rune(s[left]), &a, &b, &c, -1)
			left++
		}
	}

	return n
}

func maximumCount(nums []int) int {
	n := len(nums)
	result := 0
	left, right := 0, n

	for left < right {
		mid := (left + right) / 2
		if nums[mid] >= 0 {
			right = mid
		}
		if nums[mid] < 0 {
			left = mid + 1
		}
	}

	result = left

	left, right = 0, n

	for left < right {
		mid := (left + right) / 2
		if nums[mid] > 0 {
			right = mid
		}
		if nums[mid] <= 0 {
			left = mid + 1
		}
	}

	result = max(result, n-left)

	return result
}

// https://leetcode.com/problems/zero-array-transformation-ii/description/?envType=daily-question&envId=2025-03-13
/*
1 <= nums.length <= 10^5
0 <= nums[i] <= 5 * 10^5
1 <= queries.length <= 10^5
queries[i].length == 3
0 <= li <= ri < nums.length
1 <= vali <= 5
*/

/*
naive approach - O(n*m)
but we can do better

let's think about solution with O(n+m) or O((n+m)*log(m))?

we can use difference array and prefix sum to compute queries
this solution has O((n+m)*log(m)) complexity

we iterate queries with binary search - it's log(m)
for each set of queries:
- compute difference array in worst case scenario - m
- restore values from difference array and compare with nums - n

there is a better solution
if we iterate over nums and for each element we apply only required amount of queries
we still need to maintain difference array
but in this scenario we will iterate over nums and queries only once
this solution has O(n+m) complexity
*/

/*
2 0 2

l r v
0 2 1
0 2 1
1 1 3

sum
2 5 2
prefix sum
2 7 9

2 0 2
prefix sum
2 2 4

difference array
1 0 0 -1
prefix sum
1 1 1 0
*/
func differenceZeroArray(n int, queries [][]int) []int {
	diff := make([]int, n+1)
	for _, query := range queries {
		l, r, v := query[0], query[1], query[2]
		diff[l] += v
		diff[r+1] -= v
	}
	return diff
}

func checkZeroArray(nums []int, diff []int) bool {
	prefix := 0
	for i := range nums {
		prefix += diff[i]
		if prefix < nums[i] {
			return false
		}
	}
	return true
}

// binary search solution
func minZeroArray(nums []int, queries [][]int) int {
	n := len(nums)
	l, r := 0, len(queries)

	for l < r {
		m := (l + r) / 2
		diff := differenceZeroArray(n, queries[:m])

		if checkZeroArray(nums, diff) {
			r = m
		} else {
			l = m + 1
		}
	}

	diff := differenceZeroArray(n, queries[:l])
	if checkZeroArray(nums, diff) {
		return l
	} else {
		return -1
	}
}

// O(n)
func checkCandies(candies []int, children int, portion int) bool {
	left := children

	for _, c := range candies {
		left -= c / portion

		if left <= 0 {
			return true
		}
	}

	return false
}

// O(n)
func maxPile(candies []int) int {
	max := 0

	for _, c := range candies {
		if c > max {
			max = c
		}
	}

	return max
}

// https://leetcode.com/problems/maximum-candies-allocated-to-k-children/description/
// binary search solution
// O(n*log(m)) where m - maximum value of candies in a pile
func maximumCandies(candies []int, k int64) int {
	left, right := 1, maxPile(candies)+1

	for left < right-1 {
		mid := (left + right) / 2

		if checkCandies(candies, int(k), mid) {
			left = mid
		} else {
			right = mid
		}
	}

	if checkCandies(candies, int(k), left) {
		return left
	} else {
		return 0
	}
}

// [2, 3, 5, 9], 2, 5
// gready checks if we can rob houses
func isFeasible(nums []int, k int, value int) bool {
	robed := 0
	for i := 0; i < len(nums); {
		if nums[i] <= value {
			robed++
			i += 2
		} else {
			i += 1
		}
		if robed >= k {
			return true
		}
	}

	return false
}

// https://leetcode.com/problems/house-robber-iv/
/*
1 <= nums.length <= 10^5
1 <= nums[i] <= 10^9
1 <= k <= (nums.length + 1)/2
*/
/*
naive, brute force approach will be O(n!/k!(n-k)!)
- recursively check all posible combinations n!/k!(n-k)!
(traceback)

can we do better?

e.g.
[2, 3, 5, 9], k = 2

all posible variants:
2, 5 - 5 minimum
2, 9 - 9
3, 9 - 9

the way to solve this is - greedy algorithm to check if we can rob k houses with capacity m O(n)
binary search the lowest posible capacity m starting from min(nums), to max(nums)
*/
func minCapability(nums []int, k int) int {
	left, right := slices.Min(nums), slices.Max(nums)+1

	for left < right-1 {
		mid := (left + right) / 2
		if isFeasible(nums, k, mid) {
			right = mid
		} else {
			left = mid
		}
	}

	if isFeasible(nums, k, left) {
		return left
	}

	return right
}

// https://leetcode.com/problems/house-robber/
/*
[5,1,3,4] - 9

(p, a)
(0, 0)
(0, 5)
(5, 1)
(5, 8)
(8, 9)

dp[i] = max(dp[i-2] + nums[i], dp[i-1])
*/
func rob(nums []int) int {
	prev, last := 0, 0

	if len(nums) == 1 {
		return nums[0]
	}

	if len(nums) == 2 {
		return max(nums[0], nums[1])
	}

	prev = nums[0]
	last = max(nums[1], prev)

	for i := 2; i < len(nums); i++ {
		temp := max(prev+nums[i], last)
		prev = last
		last = temp
	}

	return last
}

// https://leetcode.com/problems/minimum-time-to-repair-cars/
/*
naive approach - increment amount of cars for smallest ranks,
then if time is comparable with greater ranks - increment amount of cars for them too

time complexity in that case - O(n*m), where n - ranks.length, m - cars

more clever approach will be constructing rank frequencies and function of
saturated mechanics occupancy, but still this leads as to O(n+m) complexity

time of repair n cars = r * n^2,
repairs by k mechanics same rank = r * (ceil(n/k))^2
repairs by mechanic with rank r, in time t = floor(sqrt(t/r))

the best we can do seems like binary search of minimal time O(n + log m), where
left boundary is minimal posible time e.g. ceil(cars / ranks.length) - assuming all mechanics 1 rank
and right boundary is maximum posible time e.g. max(ranks) * (ceil(cars/ranks.length))^2 + 1

simple binary search solution will be checking if mechanics can repair cars in time t - O(n)
overall time complexity - O(n*log m)
*/
/*
1 <= ranks.length <= 10^5
1 <= ranks[i] <= 100 -> 10^4 * 10^6 = 10^10 max time (~10^9 max int32 2 147 483 647)
1 <= cars <= 10^6
*/
func areReparableIn(ranks []int, cars int, time int64) bool {
	total := 0
	for _, r := range ranks {
		total += int(math.Sqrt(float64(time / int64(r))))
	}
	return total >= cars
}

func repairCars(ranks []int, cars int) int64 {
	n := len(ranks)
	left := int64(math.Ceil(float64(cars) / float64(n)))
	max_rank := int64(slices.Max(ranks))
	right := int64(max_rank*left*left) + 1

	for left < right-1 {
		mid := (left + right) / 2
		if areReparableIn(ranks, cars, mid) {
			right = mid
		} else {
			left = mid
		}
	}

	if areReparableIn(ranks, cars, left) {
		return left
	} else {
		return right
	}
}
