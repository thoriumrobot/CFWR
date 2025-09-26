// Test case for issue #34: https://github.com/kelloggm/checker-framework/issues/34

using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

class IndexForTwoArrays2 {

    public bool equals(int[] da1, int[] da2) {
        if (da1.Length != da2.Length) {
            return false;
        }
        int k = 0;

        for (int i = 0; i < da1.Length; i++) {
            if (da1[i] != da2[i]) {
                return false;
            }
        }
        return true;
    }
}
