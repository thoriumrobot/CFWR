// Test case for plume-lib bug fixed in 8224707


public class Duplicates {

	/*@ public normal_behavior
@ requires nums.length >= 1;
@ ensures true;
@*/
    public static int[] missing_numbers(int [] nums) {
	{ // avoid modifying parameter
	    int[] nums_copy = new int[nums.length];
	    nums = nums_copy;
	}
	int min = nums[0];
	int max = nums[nums.length-1];
	//:: error: (array.length.negative)
	int[] result = new int[max - min + 1 - nums.length];
	return result;
     }
}
