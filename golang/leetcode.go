package leetcode

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
