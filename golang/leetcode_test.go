package leetcode

import (
	"reflect"
	"testing"
)

func TestApplyOperations(t *testing.T) {
	result := applyOperations([]int{1, 2, 2, 0, 1, 1})
	expected := []int{1, 4, 2, 0, 0, 0}
	if !reflect.DeepEqual(result, expected) {
		t.Fatalf("\nresult:   %v\nexpected: %v", result, expected)
	}
}

func TestMergeArraysCase1(t *testing.T) {
	a := [][]int{{1, 2}, {2, 3}, {3, 4}}
	b := [][]int{{1, 3}, {3, 4}}
	result := mergeArrays(a, b)
	expected := [][]int{{1, 5}, {2, 3}, {3, 8}}
	if !reflect.DeepEqual(result, expected) {
		t.Fatalf("\nresult:   %v\nexpected: %v", result, expected)
	}
}

func TestMergeArraysCase2(t *testing.T) {
	a := [][]int{{1, 2}, {2, 3}, {3, 4}, {4, 5}, {7, 8}}
	b := [][]int{{1, 3}, {3, 4}, {5, 6}}
	result := mergeArrays(a, b)
	expected := [][]int{{1, 5}, {2, 3}, {3, 8}, {4, 5}, {5, 6}, {7, 8}}
	if !reflect.DeepEqual(result, expected) {
		t.Fatalf("\nresult:   %v\nexpected: %v", result, expected)
	}
}

func TestPivotArray(t *testing.T) {
	result := pivotArray([]int{10, 16, 3, 10, 13, 3, 7, 8}, 10)
	expected := []int{3, 3, 7, 8, 10, 10, 16, 13}
	if !reflect.DeepEqual(result, expected) {
		t.Fatalf("\nresult:   %v\nexpected: %v", result, expected)
	}
}

func TestCheckPowersOfThreeCase1(t *testing.T) {
	result := checkPowersOfThree(12)
	expected := true
	if result != expected {
		t.Fatalf("\nresult:   %v\nexpected: %v", result, expected)
	}
}

func TestCheckPowersOfThreeCase2(t *testing.T) {
	result := checkPowersOfThree(91)
	expected := true
	if result != expected {
		t.Fatalf("\nresult:   %v\nexpected: %v", result, expected)
	}
}

func TestCheckPowersOfThreeCase3(t *testing.T) {
	result := checkPowersOfThree(21)
	expected := false
	if result != expected {
		t.Fatalf("\nresult:   %v\nexpected: %v", result, expected)
	}
}

func TestColoredCells(t *testing.T) {
	expected := map[int]int64{
		1: 1,
		2: 5,
		3: 13,
		4: 25,
		5: 41,
		6: 61,
	}

	for n, e := range expected {
		c := coloredCells(n)
		if c != e {
			t.Fatalf("\nresult:   %v\nexpected: %v", c, e)
		}
	}
}

func TestFindMissingAndRepeatedValues(t *testing.T) {
	result := findMissingAndRepeatedValues([][]int{{1, 3}, {2, 2}})
	expected := []int{2, 4}
	if !reflect.DeepEqual(result, expected) {
		t.Fatalf("\nresult:   %v\nexpected: %v", result, expected)
	}
}

func TestClosestPrimes(t *testing.T) {
	result := closestPrimes(10, 20)
	expected := []int{11, 13}
	if !reflect.DeepEqual(result, expected) {
		t.Fatalf("\nresult:   %v\nexpected: %v", result, expected)
	}
}

func TestClosestPrimesNoPrimes(t *testing.T) {
	result := closestPrimes(4, 6)
	expected := []int{-1, -1}
	if !reflect.DeepEqual(result, expected) {
		t.Fatalf("\nresult:   %v\nexpected: %v", result, expected)
	}
}

func TestClosestPrimesSmall(t *testing.T) {
	result := closestPrimes(1, 6)
	expected := []int{2, 3}
	if !reflect.DeepEqual(result, expected) {
		t.Fatalf("\nresult:   %v\nexpected: %v", result, expected)
	}
}

func TestClosestPrimesBig(t *testing.T) {
	result := closestPrimes(710119, 710189)
	expected := []int{710119, 710189}
	if !reflect.DeepEqual(result, expected) {
		t.Fatalf("\nresult:   %v\nexpected: %v", result, expected)
	}
}

func TestIsPrime1(t *testing.T) {
	if isPrime(1) {
		t.Fatalf("expected 1 is not a prime")
	}
}

func TestIsPrime2(t *testing.T) {
	if !isPrime(2) {
		t.Fatalf("expected 2 is a prime")
	}
}

func TestIsPrime3(t *testing.T) {
	if !isPrime(3) {
		t.Fatalf("expected 3 is a prime")
	}
}

func TestIsPrime4(t *testing.T) {
	if isPrime(4) {
		t.Fatalf("expected 4 is not a prime")
	}
}

func TestIsPrime710119(t *testing.T) {
	if !isPrime(710119) {
		t.Fatalf("expected 710119 is a prime")
	}
}

func TestIsPrime710189(t *testing.T) {
	if !isPrime(710189) {
		t.Fatalf("expected 710189 is a prime")
	}
}

func TestMinimumRecolors(t *testing.T) {
	result := minimumRecolors("WBBWWBBWBW", 7)
	expected := 3
	if result != expected {
		t.Fatalf("\nresult:   %v\nexpected: %v", result, expected)
	}
}

/*
0123456
WBWBBBW

0 w=1
1 w=1, min=1
2 w=2, w=1, min=1
3 w=1, min=1
4 w=0, min=0


*/

func TestMinimumRecolors2(t *testing.T) {
	result := minimumRecolors("WBWBBBW", 2)
	expected := 0
	if result != expected {
		t.Fatalf("\nresult:   %v\nexpected: %v", result, expected)
	}
}

func TestNumberOfAlternatingGroups(t *testing.T) {
	result := numberOfAlternatingGroups([]int{0, 1, 0, 1, 0}, 3)
	expected := 3
	if result != expected {
		t.Fatalf("\nresult:   %v\nexpected: %v", result, expected)
	}
}

func TestCountOfSubstrings1(t *testing.T) {
	result := countOfSubstrings("aeioqq", 1)
	expected := int64(0)
	if result != expected {
		t.Fatalf("\nresult:   %v\nexpected: %v", result, expected)
	}
}

func TestCountOfSubstrings2(t *testing.T) {
	result := countOfSubstrings("aeiou", 0)
	expected := int64(1)
	if result != expected {
		t.Fatalf("\nresult:   %v\nexpected: %v", result, expected)
	}
}

func TestCountOfSubstrings3(t *testing.T) {
	result := countOfSubstrings("ieaouqqieaouqq", 1)
	expected := int64(3)
	if result != expected {
		t.Fatalf("\nresult:   %v\nexpected: %v", result, expected)
	}
}

func TestCountOfSubstrings4(t *testing.T) {
	result := countOfSubstrings("iqeaouqi", 2)
	expected := int64(3)
	if result != expected {
		t.Fatalf("\nresult:   %v\nexpected: %v", result, expected)
	}
}

func TestCountOfSubstrings5(t *testing.T) {
	result := countOfSubstrings("aadieuoh", 1)
	expected := int64(2)
	if result != expected {
		t.Fatalf("\nresult:   %v\nexpected: %v", result, expected)
	}
}

// aoaiuefi
//
// aoaiuef
// aoaiuefi
//
//	oaiuef
//	oaiuefi
func TestCountOfSubstrings6(t *testing.T) {
	result := countOfSubstrings("aoaiuefi", 1)
	expected := int64(4)
	if result != expected {
		t.Fatalf("\nresult:   %v\nexpected: %v", result, expected)
	}
}

func TestNumberOfSubstrings(t *testing.T) {
	result := numberOfSubstrings("abcabc")
	expected := 10
	if result != expected {
		t.Fatalf("\nresult:   %v\nexpected: %v", result, expected)
	}
}

func TestMaximumCount(t *testing.T) {
	result := maximumCount([]int{-1, 0, 0, 4, 5})
	expected := 2
	if result != expected {
		t.Fatalf("\nresult:   %v\nexpected: %v", result, expected)
	}
}

func TestMinZeroArray(t *testing.T) {
	result := minZeroArray([]int{2, 0, 2}, [][]int{{0, 2, 1}, {0, 2, 1}, {1, 1, 3}})
	expected := 2
	if result != expected {
		t.Fatalf("\nresult:   %v\nexpected: %v", result, expected)
	}
}

func TestMaximumCandies(t *testing.T) {
	result := maximumCandies([]int{5, 8, 6}, 3)
	expected := 5
	if result != expected {
		t.Fatalf("\nresult:   %v\nexpected: %v", result, expected)
	}
}

func TestMaximumCandiesMax(t *testing.T) {
	result := maximumCandies([]int{11, 11, 11}, 3)
	expected := 11
	if result != expected {
		t.Fatalf("\nresult:   %v\nexpected: %v", result, expected)
	}
}

func TestMaximumCandiesNot(t *testing.T) {
	result := maximumCandies([]int{2, 5}, 11)
	expected := 0
	if result != expected {
		t.Fatalf("\nresult:   %v\nexpected: %v", result, expected)
	}
}

func TestIsFeasible(t *testing.T) {
	result := isFeasible([]int{2, 3, 5, 9}, 2, 5)
	expected := true
	if result != expected {
		t.Fatalf("\nresult:   %v\nexpected: %v", result, expected)
	}
}

func TestIsFeasibleNo(t *testing.T) {
	result := isFeasible([]int{2, 3, 5, 9}, 2, 3)
	expected := false
	if result != expected {
		t.Fatalf("\nresult:   %v\nexpected: %v", result, expected)
	}
}

func TestItCanBeRepairedIn(t *testing.T) {
	result := areReparableIn([]int{4, 2, 3, 1}, 10, 7)
	expected := false
	if result != expected {
		t.Fatalf("\nresult:   %v\nexpected: %v", result, expected)
	}
}

func TestRepairCars(t *testing.T) {
	result := repairCars([]int{4, 2, 3, 1}, 10)
	expected := int64(16)
	if result != expected {
		t.Fatalf("\nresult:   %v\nexpected: %v", result, expected)
	}
}

func TestCountCompleteComponents(t *testing.T) {
	result := countCompleteComponents(3, [][]int{
		{0, 1}, {1, 2}, {2, 0},
	})
	expected := 1
	if result != expected {
		t.Fatalf("\nresult:   %v\nexpected: %v", result, expected)
	}
}

func TestCountPaths(t *testing.T) {
	result := countPaths(3, [][]int{
		{0, 1, 1}, {1, 2, 1}, {2, 0, 1},
	})
	expected := 1
	if result != expected {
		t.Fatalf("\nresult:   %v\nexpected: %v", result, expected)
	}
}

func TestCountDays(t *testing.T) {
	result := countDays(10, [][]int{{5, 7}, {1, 3}, {9, 10}})
	expected := 2
	if result != expected {
		t.Fatalf("\nresult:   %v\nexpected: %v", result, expected)
	}
}

func TestCheckValidLineCuts(t *testing.T) {
	result := checkValidLineCuts([][]int{{0, 1}, {1, 2}, {2, 3}, {3, 4}})
	expected := true
	if result != expected {
		t.Fatalf("\nresult:   %v\nexpected: %v", result, expected)
	}
}

func TestCheckValidCuts(t *testing.T) {
	result := checkValidCuts(5, [][]int{{1, 0, 5, 2}, {0, 2, 2, 4}, {3, 2, 5, 3}, {0, 4, 4, 5}})
	expected := true
	if result != expected {
		t.Fatalf("\nresult:   %v\nexpected: %v", result, expected)
	}
}

func TestCheckValidCuts2(t *testing.T) {
	result := checkValidCuts(5, [][]int{{0, 0, 1, 3}, {1, 0, 2, 3}, {2, 0, 3, 3}})
	expected := true
	if result != expected {
		t.Fatalf("\nresult:   %v\nexpected: %v", result, expected)
	}
}

func TestCheckValidCutsNegative(t *testing.T) {
	// [[0,2,2,4],[1,0,3,2],[2,2,3,4],[3,0,4,2],[3,2,4,4]]
	result := checkValidCuts(5, [][]int{{0, 2, 2, 4}, {1, 0, 3, 2}, {2, 2, 3, 4}, {3, 0, 4, 2}, {3, 2, 4, 4}})
	expected := false
	if result != expected {
		t.Fatalf("\nresult:   %v\nexpected: %v", result, expected)
	}
}

func TestMinOperations(t *testing.T) {
	result := minOperations([][]int{{529, 529, 989}, {989, 529, 345}, {989, 805, 69}}, 92)
	expected := 1
	if result != expected {
		t.Fatalf("\nresult:   %v\nexpected: %v", result, expected)
	}
}

func TestMinimumIndex(t *testing.T) {
	result := minimumIndex([]int{1, 2, 1})
	expected := -1
	if result != expected {
		t.Fatalf("\nresult:   %v\nexpected: %v", result, expected)
	}
}

func TestMaxPoints(t *testing.T) {
	result := maxPoints([][]int{{1, 2, 3}, {2, 5, 7}, {3, 5, 1}}, []int{5, 6, 2})
	expected := []int{5, 8, 1}
	if !reflect.DeepEqual(result, expected) {
		t.Fatalf("\nresult:   %v\nexpected: %v", result, expected)
	}

}

func TestPrimeScore(t *testing.T) {
	result := primeScore(6)
	expected := 2
	if result != expected {
		t.Fatalf("\nresult:   %v\nexpected: %v", result, expected)
	}
}

func TestMaximumScore(t *testing.T) {
	// [12,5,1,6,9,1,17,14]
	result := maximumScore([]int{12, 5, 1, 6, 9, 1, 17, 14}, 12)
	expected := 62996359
	if result != expected {
		t.Fatalf("\nresult:   %v\nexpected: %v", result, expected)
	}
}

func TestMaximumScore1(t *testing.T) {
	// [12,5,1,6,9,1,17,14]
	result := maximumScore([]int{12, 5, 1, 6, 9, 1, 17, 14}, 12)
	expected := 62996359
	if result != expected {
		t.Fatalf("\nresult:   %v\nexpected: %v", result, expected)
	}
}
