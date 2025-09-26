using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

public class ArrayAssignmentSameLen {

    private readonly int[] i_array;
    private readonly int i_index;

    [ContractInvariantMethod]
    private void Invariant()
    {
        Contract.Invariant(i_index >= 0 && i_index < i_array.Length);
    }

    ArrayAssignmentSameLen(int[] array, int index) {
        Contract.Requires(index >= 0 && index < array.Length);
        i_array = array;
        i_index = index;
    }

    void test1(int[] a, int[] b, int index) {
        Contract.Requires(index <= a.Length);
        int[] array = a;
        
        // :: error: (assignment.type.incompatible)
        int i = index;
        if(TestHelper.nondet()) Contract.Assert(i + 0 < array.Length && i + -3 < b.Length);
    }

    void test2(int[] a, int[] b, int i) {
        Contract.Requires(i < a.Length);
        int[] c = a;
        // :: error: (assignment.type.incompatible)
        int x = i;
        if(TestHelper.nondet()) Contract.Assert(x < c.Length && x < b.Length);
        int y = i;
        if(TestHelper.nondet()) Contract.Assert(y < c.Length);
    }

    void test3(int[] a, int i, int x) {
        Contract.Requires(i < a.Length);
        Contract.Requires(x >= 0);
        int[] c1 = a;
        // See useTest3 for an example of why this assignment should fail.

        // :: error: (assignment.type.incompatible)
        int z = i;
        if(TestHelper.nondet()) Contract.Assert(z + 0 < c1.Length && z + x < c1.Length);        
    }

    void test4(
            int[] a,
                    int i,
            int x) {
        Contract.Requires(i + 0 < a.Length && i + x < a.Length);
        Contract.Requires(x >= 0);
        int[] c1 = a;
        
        int z = i;
        if(TestHelper.nondet()) Contract.Assert(z + 0 < c1.Length && z + x < c1.Length);        
    }

    void useTest3() {
        int[] a = {1, 3};
        test3(a, 0, 10);
    }
}
