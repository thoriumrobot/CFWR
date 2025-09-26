using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

public class MinLenOneAndLength {
    public void m1(int [] a, int[] b) {
        Contract.Requires(a.Length >= 1);
        int i = a.Length / 2;
        if(TestHelper.nondet()) Contract.Assert(i >= 0 && i < a.Length);
        // :: error: (assignment.type.incompatible)
        int j = b.Length / 2;
        if(TestHelper.nondet()) Contract.Assert(j >= 0 && j < b.Length);
    }
}
