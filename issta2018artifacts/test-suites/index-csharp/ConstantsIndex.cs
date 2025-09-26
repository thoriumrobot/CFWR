using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

class ConstantsIndex {

    void test() {
        int[] arr = {1, 2, 3};
        if(TestHelper.nondet()) Contract.Assert(arr.Length >= 3);
        int i = arr[1];
        // :: error: (array.access.unsafe.high.constant)
        int j = arr[3];
    }
}
