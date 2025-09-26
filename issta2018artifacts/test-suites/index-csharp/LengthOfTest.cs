using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

class LengthOfTest {
    void foo(int[] a, int x) {
        Contract.Requires(x == a.Length);
        int y = x;
        if(TestHelper.nondet()) Contract.Assert(y >= 0 && y <= a.Length);
        // :: error: (assignment.type.incompatible)
        int w = x;
        if(TestHelper.nondet()) Contract.Assert(w >= 0 && w < a.Length);
        int z = a.Length;
        if(TestHelper.nondet()) Contract.Assert(z >= 0 && z == a.Length);
    }
}
