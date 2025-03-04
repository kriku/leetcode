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
