using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

class ArrayCopy {

    void copy(int [] nums) {
        Contract.Requires(nums.Length >= 1);
        int[] nums_copy = new int[nums.Length];
        Array.Copy(nums, 0, nums_copy, 0, nums.Length);
        nums = nums_copy;
    }
}
