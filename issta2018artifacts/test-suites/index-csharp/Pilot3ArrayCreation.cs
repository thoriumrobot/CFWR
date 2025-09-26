// This test case is for issue 44: https://github.com/kelloggm/checker-framework/issues/44

using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

class Pilot3ArrayCreation {
    void test(int[] firstArray, int[][] secondArray) {
        int[] newArray = new int[firstArray.Length + secondArray.Length];
        for (int i = 0; i < firstArray.Length; i++) {
            newArray[i] = firstArray[i]; // or newArray[i] = secondArray[i];
        }
    }
}
