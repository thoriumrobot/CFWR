using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

class Issue21 {

    void test(int[] arr, int[] arr2) {
        for (int i = 0; i < arr2.Length && i < arr.Length; i++) {
            arr[i] = arr2[i];
        }
    }
}
