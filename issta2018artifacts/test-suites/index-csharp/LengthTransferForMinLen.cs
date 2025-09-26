using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

class LengthTransferForMinLen {
    void exceptional_control_flow(int[] a) {
        if (a.Length == 0) {
            throw new ArgumentException();
        }
        int[] b = a;
        if(TestHelper.nondet()) Contract.Assert(b.Length >= 1);
    }

    void equal_to_return(int[] a) {
        if (a.Length == 0) {
            return;
        }
        int[] b = a;
        if(TestHelper.nondet()) Contract.Assert(b.Length >= 1);
    }

    void gt_check(int[] a) {
        if (a.Length > 0) {
            int[] b = a;
            if(TestHelper.nondet()) Contract.Assert(b.Length >= 1);
        }
    }
}
