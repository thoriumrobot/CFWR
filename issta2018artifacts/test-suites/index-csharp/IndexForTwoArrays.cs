using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

class IndexForTwoArrays {

    public int compare(double[] a1, double[] a2) {
        if (a1 == a2) {
            return 0;
        }
        int len = Math.Min(a1.Length, a2.Length);
        for (int i = 0; i < len; i++) {
            if (a1[i] != a2[i]) {
                return ((a1[i] > a2[i]) ? 1 : -1);
            }
        }
        return a1.Length - a2.Length;
    }
}
