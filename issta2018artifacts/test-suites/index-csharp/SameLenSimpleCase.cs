using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

class SameLenSimpleCase {
    public int compare(int[] a1, int[] a2) {
        if (a1.Length != a2.Length) {
            return a1.Length - a2.Length;
        }
        for (int i = 0; i < a1.Length; i++) {
            if (a1[i] != a2[i]) {
                return ((a1[i] > a2[i]) ? 1 : -1);
            }
        }
        return 0;
    }
}
