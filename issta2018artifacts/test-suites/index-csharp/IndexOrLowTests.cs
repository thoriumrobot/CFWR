using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

public class IndexOrLowTests {
    int[] array = {1, 2};

    int index = -1;

    [ContractInvariantMethod]
    private void Invariant()
    {
        Contract.Invariant(index >= -1 && index < array.Length);
    }

    void test() {

        if (index != -1) {
            array[index] = 1;
        }

        int y = index + 1;

        if(TestHelper.nondet()) Contract.Assert(y >= 0 && y <= array.Length);
        // :: error: (array.access.unsafe.high)
        array[y] = 1;
        if (y < array.Length) {
            array[y] = 1;
        }
        // :: error: (assignment.type.incompatible)
        index = array.Length;
    }

    void test2(int param) {
        Contract.Requires(param >= -1);
        Contract.Requires(param < array.Length);
        index = array.Length - 1;
        int x = index;
        if(TestHelper.nondet()) Contract.Assert(x >= -1);
        if(TestHelper.nondet()) Contract.Assert(x < array.Length);
        index = param;
    }
}
