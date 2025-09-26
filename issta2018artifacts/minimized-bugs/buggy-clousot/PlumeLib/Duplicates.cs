using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Linq;
using System.Text;
using System.Threading.Tasks;


public class Duplicates
{
    public static int[] missing_numbers(int[] nums) {
        Contract.Requires(nums.Length >= 1);
	    { // avoid modifying parameter
	        int[] nums_copy = new int[nums.Length];
            Array.Copy(nums, 0, nums_copy, 0, nums.Length);
	        nums = nums_copy;
	    }
        Array.Sort(nums);
	    int min = nums[0];
        int max = nums[nums.Length - 1];
        //:: error: (array.length.negative)
        int[] result = new int[max - min + 1 - nums.Length];
	    return result;
    }
}