using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

class ArrayLength {
    void test() {
        int[] arr = {1, 2, 3};
        int a = arr.Length;
        if(TestHelper.nondet()) Contract.Assert(a <= arr.Length);
    }
}
