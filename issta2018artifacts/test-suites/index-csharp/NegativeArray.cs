using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

public class NegativeArray {

    public static void negativeArray(int len) {
        Contract.Requires(len >= -1);
        // :: error: (array.Length.negative)
        int[] arr = new int[len];
    }
}
