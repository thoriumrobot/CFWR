using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

public class IntroSub {
    void test(int[] arr) {
        // :: error: (assignment.type.incompatible)
        int a = 3;
        if(TestHelper.nondet()) Contract.Assert(a < arr.Length);
        // :: error: (assignment.type.incompatible)
        int c = a - (-1);
        if(TestHelper.nondet()) Contract.Assert(c < arr.Length);
        int c1 = a - (-1);
        if(TestHelper.nondet()) Contract.Assert(c1 <= arr.Length);
        int d = a - 0;
        if(TestHelper.nondet()) Contract.Assert(d < arr.Length);
        int e = a - 7;
        if(TestHelper.nondet()) Contract.Assert(e < arr.Length);
        // :: error: (assignment.type.incompatible)
        int f = a - (-7);
        if(TestHelper.nondet()) Contract.Assert(f < arr.Length);

        // :: error: (assignment.type.incompatible)
        int j = 7;
        if(TestHelper.nondet()) Contract.Assert(j <= arr.Length);
    }
}
