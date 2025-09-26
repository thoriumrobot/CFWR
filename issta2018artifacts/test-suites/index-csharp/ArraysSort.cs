using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

class ArraysSort {

    void sortInt(int [] nums) {
        Contract.Requires(nums.Length >= 10);
        // Checks the correct handling of the toIndex parameter
        Array.Sort(nums, 0, 10);
        // :: error: (argument.type.incompatible)
        Array.Sort(nums, 0, 11);
    }
}
