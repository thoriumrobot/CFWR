using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

/*@SuppressWarnings("lowerbound")*/
class ArrayIntro {
    void test() {
        int[] arr = new int[5];
        if(TestHelper.nondet()) Contract.Assert(arr.Length >= 5);
        int a = 9;
        a += 5;
        a -= 2;
        int[] arr1 = new int[a];
        if(TestHelper.nondet()) Contract.Assert(arr1.Length >= 12);
        int[] arr2 = {1, 2, 3};
        if(TestHelper.nondet()) Contract.Assert(arr2.Length >= 3);
        // :: error: (assignment.type.incompatible)
        int[] arr3 = {4, 5, 6};
        if(TestHelper.nondet()) Contract.Assert(arr3.Length >= 4);
        // :: error: (assignment.type.incompatible)
        int[] arr4 = new int[4];
        if(TestHelper.nondet()) Contract.Assert(arr4.Length >= 7);
        // :: error: (assignment.type.incompatible)
        int[] arr5 = new int[a];
        if(TestHelper.nondet()) Contract.Assert(arr5.Length >= 16);
    }
}
