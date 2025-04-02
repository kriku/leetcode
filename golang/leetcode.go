package leetcode

import (
	"container/heap"
	"maps"
	"math"
	"slices"
	"sort"
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

// https://leetcode.com/problems/divide-array-into-equal-pairs/
func divideArray(nums []int) bool {
	pairs := make([]bool, 501)

	for _, n := range nums {
		pairs[n] = !pairs[n]
	}

	canBeDivided := true

	for _, p := range pairs {
		canBeDivided = canBeDivided && !p
	}

	return canBeDivided
}

// https://leetcode.com/problems/longest-nice-subarray/
/*

1 - 1        // we need to "pop" this from xor and from bitwise and
2 - 10
4 - 100    // bitwise xor is 111
1 - 1      // bitwise and is 1, bitwise xor is 110
8 - 1000

another case

1 - 1        // we need to "pop" this from xor and from bitwise and
2 - 10       // and also pop this one
4 - 100    // bitwise xor is 111
3 - 11     // bitwise and is 11, bitwise xor is 100
8 - 1000

                &       ^
1  - 1          1       1
3  - 11         1       10
8  - 100
48 - 110000
10 - 1010

here we should use definitely sliding window approach

 [1, 2, 4, 1, 8]
^ starting from 0

then starting to move either left or right boundary

we can extend right boundary as far as we have next number bitwise and with whole xor equals 0
 [1, 2, 4, 1, 8]
  ^ xor is 0, 1 & 0 == 0 - we already have right boundary on 0
     ^ xor is 1, and is 0, 1 & 10 == 0 - we can extend right boundary +1
        ^ xor is 11, and is 0, ...
           ^ xor is 111, and is 0, 1 & 111 == 1 - we cannot extend right boundary any more right should stay at 3
  ^ xor is 110, and is 1 -
*/
func longestNiceSubarray(nums []int) int {
	left, right, result := 0, 0, 0
	xor := 0

	for right < len(nums) && left < len(nums) {
		if xor&nums[right] == 0 {
			xor ^= nums[right]
			right++
		} else {
			xor ^= nums[left]
			left++
		}

		result = max(result, right-left)
	}

	return result
}

// https://leetcode.com/problems/minimum-cost-walk-in-weighted-graph/
func minimumCost(n int, edges [][]int, query [][]int) []int {
	adjacencyList := make([][][]int, n)
	for _, e := range edges {
		from, to, weight := e[0], e[1], e[2]
		adjacencyList[from] = append(adjacencyList[from], []int{to, weight})
		adjacencyList[to] = append(adjacencyList[to], []int{from, weight})
	}

	visited := make([]bool, n)
	components := make([]int, n)
	componentId := 0
	costs := make([]int, n)

	for i := range n {
		costs[i] = math.MaxInt32
	}

	// bfs
	for i := range n {
		if visited[i] {
			continue
		}

		toVisit := []int{i}

		for len(toVisit) > 0 {
			current := toVisit[0]
			toVisit = toVisit[1:]

			visited[current] = true
			components[current] = componentId

			for _, edge := range adjacencyList[current] {
				to, weight := edge[0], edge[1]
				if !visited[to] {
					toVisit = append(toVisit, to)
				}
				costs[componentId] &= weight
			}
		}

		componentId++
	}

	result := make([]int, len(query))
	for i, q := range query {
		from, to := q[0], q[1]
		if components[from] != components[to] {
			result[i] = -1
		} else {
			result[i] = costs[components[from]]
		}
	}

	return result
}

// https://leetcode.com/problems/find-all-possible-recipes-from-given-supplies/
func findAllRecipes(recipes []string, ingredients [][]string, supplies []string) []string {
	hasNewSupplies := true
	suppliesSet := make(map[string]bool)
	visitedRecipes := make(map[string]bool)
	doneRecipes := make([]string, 0)

	for _, s := range supplies {
		suppliesSet[s] = true
	}

	for hasNewSupplies {
		hasNewSupplies = false

		// bfs
		for i, r := range recipes {
			if visitedRecipes[r] {
				continue
			}

			isPossible := true

			for _, ingredient := range ingredients[i] {
				isPossible = isPossible && suppliesSet[ingredient]
			}

			if isPossible {
				visitedRecipes[r] = true
				hasNewSupplies = true
				doneRecipes = append(doneRecipes, r)
				suppliesSet[r] = true
			}
		}
	}

	return doneRecipes
}

// https://leetcode.com/problems/count-the-number-of-complete-components/
/*

connected component vertex and edges counts:

v - e
1 - 0
2 - 1
3 - 3
4 - 6
5 - 10
6 - 16
7 - 23

e(v) = e(v-1) + (v-1) - seems right

but can we express not recursively? I think we can

e(1) = e(0) + 0
e(2) = e(1) + 1 = e(0) + 0 + 1
e(3) = e(2) + 2 = e(1) + 1 + 2 = e(0) + 0 + 1 + 2
e(3) = e(3) + 3 = e(2) + 2 + 3 = e(1) + 1 + 2 + 3 = e(0) + 0 + 1 + 2 + 3

this is arithmetic sequence with sum n(n-1)/2

...

*/
func countCompleteComponents(n int, edges [][]int) int {
	completed := 0

	graph := make(map[int][]int)

	for _, e := range edges {
		from, to := e[0], e[1]
		graph[from] = append(graph[from], to)
		graph[to] = append(graph[to], from)
	}

	visited := make(map[int]bool)

	for v := range graph {
		if visited[v] {
			continue
		}

		connected := 0
		connectedEdges := 0

		toVisit := []int{v}

		for len(toVisit) > 0 {
			current := toVisit[0]
			toVisit = toVisit[1:]
			visited[current] = true
			connected++
			connectedEdges += len(graph[current])

			for _, to := range graph[current] {
				if visited[to] {
					continue
				}

				toVisit = append(toVisit, to)
				visited[to] = true
			}
		}

		if connected*(connected-1) == connectedEdges {
			completed++
		}
	}

	return completed
}

// https://leetcode.com/problems/number-of-ways-to-arrive-at-destination/
type N struct {
	value     int
	distance  int64
	neighbors []int
	distances []int
	// index in priority queue
	index int
}

// A PriorityQueue implements heap.Interface and holds Items.
type PQ []*N

func (pq PQ) Len() int { return len(pq) }

func (pq PQ) Less(i, j int) bool {
	return pq[i].distance < pq[j].distance
}

func (pq PQ) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
	pq[i].index = i
	pq[j].index = j
}

func (pq *PQ) Push(x any) {
	n := len(*pq)
	item := x.(*N)
	item.index = n
	*pq = append(*pq, item)
}

func (pq *PQ) Pop() any {
	old := *pq
	n := len(old)
	item := old[n-1]
	old[n-1] = nil  // don't stop the GC from reclaiming the item eventually
	item.index = -1 // for safety
	*pq = old[0 : n-1]
	return item
}

// update modifies the priority and value of an Item in the queue.
func (pq *PQ) update(item *N, distance int64) {
	item.distance = distance
	heap.Fix(pq, item.index)
}

func countPaths(n int, roads [][]int) int {
	graph := make([]*N, n)

	for i := range n {
		graph[i] = &N{
			value:     i,
			distance:  math.MaxInt64,
			neighbors: make([]int, 0),
			index:     i,
		}
	}

	// build graph adjacency list
	for _, e := range roads {
		from, to, distance := e[0], e[1], e[2]
		graph[from].neighbors = append(graph[from].neighbors, to)
		graph[to].neighbors = append(graph[to].neighbors, from)
		graph[from].distances = append(graph[from].distances, distance)
		graph[to].distances = append(graph[to].distances, distance)
	}

	pq := make(PQ, n)
	graph[0].distance = 0
	for i := range n {
		pq[i] = graph[i]
	}

	heap.Init(&pq)

	ways := make([]int, n)
	ways[0] = 1
	current := 0

	for pq.Len() > 0 {
		node := heap.Pop(&pq).(*N)
		current = node.value

		for i, neighbor := range node.neighbors {
			distance := node.distance + int64(node.distances[i])

			if graph[neighbor].distance > distance {
				pq.update(graph[neighbor], distance)
				ways[neighbor] = ways[current]
			} else if graph[neighbor].distance == distance {
				ways[neighbor] = (ways[neighbor] + ways[current]) % 1000000007
			}
		}
	}

	return ways[n-1]
}

// we can use difference map to solve this
func countDays(days int, meetings [][]int) int {
	difference := make(map[int]int)
	previous := days

	for _, m := range meetings {
		previous = min(previous, m[0])
		difference[m[0]]++
		difference[m[1]+1]--
	}

	current := 0
	free := previous - 1
	dates := slices.Sorted(maps.Keys(difference))

	for _, date := range dates {
		if current == 0 {
			free += date - previous
		}
		current += difference[date]
		previous = date
	}

	free += days - previous + 1

	return free
}

// https://leetcode.com/problems/check-if-grid-can-be-cut-into-sections/

type Segments [][]int

func (s Segments) Len() int           { return len(s) }
func (s Segments) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }
func (s Segments) Less(i, j int) bool { return s[i][0] < s[j][0] }

func checkValidLineCuts(s Segments) bool {
	sort.Sort(s)

	end := 0
	cuts := -1

	for _, x := range s {
		if end <= x[0] {
			cuts++
		}
		end = max(end, x[1])

		if cuts >= 2 {
			return true
		}
	}
	return false
}

func checkValidCuts(n int, rectangles [][]int) bool {
	xSegments := make(Segments, len(rectangles))
	ySegments := make(Segments, len(rectangles))

	for i, r := range rectangles {
		xSegments[i] = []int{r[0], r[2]}
		ySegments[i] = []int{r[1], r[3]}
	}

	return checkValidLineCuts(xSegments) || checkValidLineCuts(ySegments)
}

// https://leetcode.com/problems/minimum-operations-to-make-a-uni-value-grid/
func minOperations(grid [][]int, x int) int {
	flatten := make([]int, 0)
	for _, row := range grid {
		flatten = append(flatten, row...)
	}

	sort.Ints(flatten)

	to := flatten[len(flatten)/2]

	count := 0

	for _, v := range flatten {
		diff := v - to
		if diff < 0 {
			diff = -diff
		}
		if diff%x != 0 {
			return -1
		}
		count += diff / x
	}

	return count
}

// https://leetcode.com/problems/minimum-index-of-a-valid-split/
func minimumIndex(nums []int) int {
	frequency := make(map[int]int)
	fs := make([][]int, len(nums))
	f := 0
	d := 0

	for i, n := range nums {
		frequency[n]++
		if f < frequency[n] && frequency[n]*2 > i+1 {
			f = frequency[n]
			d = n
		} else {
			if f*2 <= i+1 {
				f = 0
				d = 0
			}
		}
		fs[i] = []int{d, f}
	}

	for i, fx := range fs[:len(fs)-1] {
		if fx[0] == d && (f-fx[1])*2 > len(nums)-i-1 {
			return i
		}
	}

	return -1
}

// https://leetcode.com/problems/maximum-number-of-points-from-grid-queries/
type Node struct {
	value, i, j int
}

type PriorityQueue []*Node

func (pq PriorityQueue) Len() int { return len(pq) }

func (pq PriorityQueue) Less(i, j int) bool {
	return pq[i].value < pq[j].value
}

func (pq PriorityQueue) Swap(i, j int) {
	pq[i], pq[j] = pq[j], pq[i]
}

func (pq *PriorityQueue) Push(x any) {
	item := x.(*Node)
	*pq = append(*pq, item)
}

func (pq *PriorityQueue) Pop() any {
	old := *pq
	n := len(old)
	item := old[n-1]
	old[n-1] = nil // don't stop the GC from reclaiming the item eventually
	*pq = old[0 : n-1]
	return item
}

type Query struct {
	index, value int
}

type Queries []*Query

func (s Queries) Len() int           { return len(s) }
func (s Queries) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }
func (s Queries) Less(i, j int) bool { return s[i].value < s[j].value }

func maxPoints(grid [][]int, queries []int) []int {
	queriesList := make(Queries, len(queries))
	queriesResults := make([]int, len(queries))

	for i, q := range queries {
		queriesList[i] = &Query{index: i, value: q}
	}

	sort.Sort(queriesList)

	isInQueue := make([][]bool, len(grid))
	for i := range isInQueue {
		isInQueue[i] = make([]bool, len(grid[0]))
	}
	toVisit := make(PriorityQueue, 0)
	toVisit.Push(&Node{grid[0][0], 0, 0})
	isInQueue[0][0] = true

	cellsVisited := 0
	for _, q := range queriesList {

		for len(toVisit) > 0 && toVisit[0].value < q.value {
			cellsVisited++

			node := heap.Pop(&toVisit).(*Node)

			if node.i+1 < len(grid) && !isInQueue[node.i+1][node.j] {
				isInQueue[node.i+1][node.j] = true
				heap.Push(&toVisit, &Node{grid[node.i+1][node.j], node.i + 1, node.j})
			}
			if node.j+1 < len(grid[0]) && !isInQueue[node.i][node.j+1] {
				isInQueue[node.i][node.j+1] = true
				heap.Push(&toVisit, &Node{grid[node.i][node.j+1], node.i, node.j + 1})
			}
			if node.i-1 >= 0 && !isInQueue[node.i-1][node.j] {
				isInQueue[node.i-1][node.j] = true
				heap.Push(&toVisit, &Node{grid[node.i-1][node.j], node.i - 1, node.j})
			}
			if node.j-1 >= 0 && !isInQueue[node.i][node.j-1] {
				isInQueue[node.i][node.j-1] = true
				heap.Push(&toVisit, &Node{grid[node.i][node.j-1], node.i, node.j - 1})
			}
		}

		queriesResults[q.index] = cellsVisited
	}

	return queriesResults
}

// https://leetcode.com/problems/apply-operations-to-maximize-score/
type Pair struct {
	index, value int
}

const MOD = 1_000_000_007

// An IntHeap is a max-heap of ints.
type IntHeap []*Pair

func (h IntHeap) Len() int           { return len(h) }
func (h IntHeap) Less(i, j int) bool { return h[i].value > h[j].value }
func (h IntHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }

func (h *IntHeap) Push(x any) {
	// Push and Pop use pointer receivers because they modify the slice's length,
	// not just its contents.
	*h = append(*h, x.(*Pair))
}

func (h *IntHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[0 : n-1]
	return x
}

func primeScore(n int) int {
	p := 0
	for i := 2; i <= int(math.Sqrt(float64(n))); i++ {
		if n%i == 0 {
			p++
			for n%i == 0 {
				n = n / i
			}
		}
	}

	if n >= 2 {
		p++
	}

	return p
}

/*
nums = []int{60, 15, 420, 2, 40}
scores = []int{3, 2, 4, 1, 2}
right = []int{5, 5, 5, 5, 5}
left = []int{-1, 0, -1, -1, -1}

let's build associated arrays
of nearest elements with bigger prime score
to the right and left of current element

to do so we will use monotonic stack

we start with arrays initialized to boundaries of the array, -1 and len(nums)
right = []int{5, 5, 5, 5, 5}
left = []int{-1, -1, -1, -1, -1}

then for each element of prime scores array we create pair of its index and value
and push it to the monotonic stack
we will pop elements from the stack, if current element is greater than the last element in the stack

// 0
ms = []*Pair{Pair{3, 0}}

// 1
ms = []*Pair{Pair{3, 0}, Pair{2, 1}}
// at this point we know, that element at index 1
// has prime score lower than element at index 0,
// so we can put index 0, to the "left" array at index 1

// if scores at index 1 will be equals to element at index 0
// e.g. scores = []int{3, 3, 2, 4, 2}
// 0 - Pair{3, 0}
ms = []*Pair{Pair{3, 0}}
right = []int{5, 5, 5, 5, 5}
left = []int{-1, -1, -1, -1, -1}
// 1 - Pair{3, 1}
ms = []*Pair{Pair{3, 0}, Pair{3, 1}}
right = []int{5, 5, 5, 5, 5}
left = []int{-1, 0, -1, -1, -1}
// 2 - Pair{2, 2}
ms = []*Pair{Pair{3, 0}, Pair{3, 1}, Pair{2, 2}}
right = []int{5, 5, 5, 5, 5}
left = []int{-1, 0, 1, -1, -1}
// 3 - Pair{4, 3}
ms = []*Pair{Pair{4, 3}}
Pair{2, 2}
Pair{3, 1}
Pair{3, 0}
right = []int{3, 3, 3, 5, 5}
left = []int{-1, 0, 1, -1, -1}
// 4 - Pair{2, 4}
ms = []*Pair{Pair{4, 3}, Pair{2, 4}}
right = []int{3, 3, 3, 5, 5}
left = []int{-1, 0, 1, -1, 3}

// in that case we will put index 0 to the "left" array at index 1
// but also we should pop index 0 from the monotonic stack
// and push Pair {3, 1} as new first biggest prime score to the "left"

// 2 etc...
*/
func maximumScore(nums []int, k int) int {
	scores := make([]int, len(nums))
	ms := make([]*Pair, 0)
	h := make(IntHeap, 0)
	heap.Init(&h)

	for i, n := range nums {
		scores[i] = primeScore(n)
		heap.Push(&h, &Pair{i, n})
	}

	left := make([]int, len(nums))
	right := make([]int, len(nums))
	for i := range left {
		left[i] = -1
		right[i] = len(nums)
	}

	for i, n := range scores {
		for len(ms) > 0 && n > ms[len(ms)-1].value {
			right[ms[len(ms)-1].index] = i
			ms = ms[:len(ms)-1]
		}

		if len(ms) > 0 {
			left[i] = ms[len(ms)-1].index
		}

		ms = append(ms, &Pair{i, n})
	}

	arrays := make([]int64, len(nums))

	for i := range nums {
		arrays[i] = int64((i - left[i]) * (right[i] - i))
	}

	sum := uint64(1)

	for k > 0 && len(h) > 0 {
		n := heap.Pop(&h).(*Pair)
		applied := min(int64(k), arrays[n.index])
		k -= int(applied)

		sum = (sum * power(uint64(n.value), int(applied))) % MOD
	}

	return int(sum)
}

func power(base uint64, exponent int) uint64 {
	result := uint64(1)

	for exponent > 0 {
		if exponent%2 == 1 {
			result = (result * base) % MOD
		}

		base = (base * base) % MOD

		exponent = exponent / 2
	}

	return result
}

// https://leetcode.com/problems/put-marbles-in-bags/
func putMarbles(weights []int, k int) int64 {
	n := len(weights)
	pairs := make([]int, n-1)
	for i := 0; i < n-1; i++ {
		pairs[i] = weights[i] + weights[i+1]
	}
	sort.Ints(pairs)
	diff := int64(0)

	// radius := min(k-1, n/2)
	for i := 0; i < k-1; i++ {
		diff += int64(pairs[n-i-1] - pairs[i])
	}

	return diff
}
