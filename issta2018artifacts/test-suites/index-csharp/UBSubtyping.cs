using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

public class UBSubtyping {
    int[] arr = {1};
    int[] arr2 = {1};
    int[] arr3 = {1};

    void test(int test) {
        Contract.Requires(test <= arr.Length && test <= arr2.Length && test <= arr3.Length);

        // :: error: (assignment.type.incompatible)
        int a = 1;
        if(TestHelper.nondet()) Contract.Assert(a <= arr.Length);
        // :: error: (assignment.type.incompatible)
        int a1 = 1;
        if(TestHelper.nondet()) Contract.Assert(a1 < arr.Length);

        // :: error: (assignment.type.incompatible)
        int b = a;
        if(TestHelper.nondet()) Contract.Assert(b < arr.Length);
        int d = a;
        if(TestHelper.nondet()) Contract.Assert(true);

        // :: error: (assignment.type.incompatible)
        int g = a;
        if(TestHelper.nondet()) Contract.Assert(g < arr2.Length);

        // :: error: (assignment.type.incompatible)
        int h = 2;
        if(TestHelper.nondet()) Contract.Assert(h <= arr.Length && h <= arr2.Length && h <= arr3.Length);
        int h2 = test;
        if(TestHelper.nondet()) Contract.Assert(h2 <= arr.Length && h2 <= arr2.Length);
        int i = test;
        if(TestHelper.nondet()) Contract.Assert(i <= arr.Length);
        int j = test;
        if(TestHelper.nondet()) Contract.Assert(j <= arr.Length && j <= arr3.Length);
    }
}