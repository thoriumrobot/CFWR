using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

class LengthTransfer {
    void exceptional_control_flow(int[] a) {
        if (a.Length == 0) {
            throw new ArgumentException();
        }
        int i = a[0];
    }

    void equal_to_return(int[] a) {
        if (a.Length == 0) {
            return;
        }
        int i = a[0];
    }

    void gt_check(int[] a) {
        if (a.Length > 0) {
            int i = a[0];
        }
    }
}
