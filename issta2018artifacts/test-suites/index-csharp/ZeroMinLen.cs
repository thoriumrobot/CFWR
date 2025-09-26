using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

public class ZeroMinLen {

    protected int [] nums;
    protected int[] nums2;

    int current_index;

    int current_index2;

    void test() {
        current_index = 0;
        // :: error: (assignment.type.incompatible)
        current_index2 = 0;
    }


    [ContractInvariantMethod]
    private void Invariant()
    {
        Contract.Invariant(nums.Length >= 1);
        Contract.Invariant(current_index >= 0 && current_index < nums.Length);
        Contract.Invariant(current_index2 >= 0 && current_index2 < nums2.Length);
    }
}
