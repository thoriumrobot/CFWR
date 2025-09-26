using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

class OneLTL {
    public static bool sorted(int[] a) {
        for (int i = 0; i < a.Length - 1; i++) {
            if (a[i + 1] < a[i]) {
                return false;
            }
        }
        return true;
    }
}
