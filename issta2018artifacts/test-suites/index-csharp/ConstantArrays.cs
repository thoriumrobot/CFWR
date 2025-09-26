using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

class ConstantArrays {
    void basic_test() {
        int[] b = new int[4];
        int[] a = {0, 1, 2, 3};
        if(TestHelper.nondet()) Contract.Assert(Contract.ForAll(0, a.Length, i => a[i] < b.Length));

        // :: error: (array.initializer.type.incompatible)::error: (assignment.type.incompatible)
        int[] a1 = {0, 1, 2, 4};
        if(TestHelper.nondet()) Contract.Assert(Contract.ForAll(0, a1.Length, i => a1[i] < b.Length));

        int[] c = {-1, 4, 3, 1};
        if(TestHelper.nondet()) Contract.Assert(Contract.ForAll(0, c.Length, i => c[i] <= b.Length));

        // :: error: (array.initializer.type.incompatible)::error: (assignment.type.incompatible)
        int[] c2 = {-1, 4, 5, 1};
        if(TestHelper.nondet()) Contract.Assert(Contract.ForAll(0, c2.Length, i => c2[i] <= b.Length));
    }

    void offset_test() {
        int[] b = new int[4];
        int[] b2 = new int[10];
        
        int[] a = {2, 3, 0};
        if(TestHelper.nondet()) Contract.Assert(Contract.ForAll(0, a.Length, i=>a[i] + -2 < b.Length && a[i] + 5 < b2.Length));

        // :: error: (array.initializer.type.incompatible)::error: (assignment.type.incompatible)
        int[] a2 = {2, 3, 5};
        if(TestHelper.nondet()) Contract.Assert(Contract.ForAll(0, a2.Length, i => a2[i] + -2 < b.Length && a2[i] + 5 < b2.Length));

        // Non-constant offsets don't work correctly. See kelloggm#120.
    }
}
