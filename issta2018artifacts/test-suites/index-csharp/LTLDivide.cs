using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

class LTLDivide {
    int[] test(int[] array) {
        //        @LTLengthOf("array") int len = array.Length / 2;
        int len = array.Length / 2;
        int[] arr = new int[len];
        for (int a = 0; a < len; a++) {
            arr[a] = array[a];
        }
        return arr;
    }

    void test2(int[] array) {
        int len = array.Length;
        int lenM1 = array.Length - 1;
        int lenP1 = array.Length + 1;
        // :: error: (assignment.type.incompatible)
        int x = len / 2;
        if(TestHelper.nondet()) Contract.Assert(x < array.Length);
        int y = lenM1 / 3;
        if(TestHelper.nondet()) Contract.Assert(y < array.Length);
        int z = len / 1;
        if(TestHelper.nondet()) Contract.Assert(z <= array.Length);
        // :: error: (assignment.type.incompatible)
        int w = lenP1 / 2;
        if(TestHelper.nondet()) Contract.Assert(w < array.Length);
    }
}
