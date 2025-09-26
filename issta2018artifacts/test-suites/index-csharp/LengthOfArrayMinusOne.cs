using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

class LengthOfArrayMinusOne {
    void test(int[] arr) {
        // :: error: (array.access.unsafe.low)
        int i = arr[arr.Length - 1];

        if (arr.Length > 0) {
            int j = arr[arr.Length - 1];
        }
    }
}
