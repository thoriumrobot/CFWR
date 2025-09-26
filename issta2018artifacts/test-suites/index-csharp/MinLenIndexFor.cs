using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

public class MinLenIndexFor {
    int[] arrayLen2 = {0, 1, 2};

    [ContractInvariantMethod]
    private void Invariant()
    {
        Contract.Invariant(arrayLen2.Length >= 2);
        Contract.Invariant(arrayLen4.Length >= 4);
    }

    void test( int i) {
        Contract.Requires(i >= 0 && i<this.arrayLen2.Length);
        int j = arrayLen2[i];
        int j2 = arrayLen2[1];
    }

    void callTest(int x) {
        test(0);
        test(1);
        // :: error: (argument.type.incompatible)
        test(2);
        // :: error: (argument.type.incompatible)
        test(3);
        test(arrayLen2.Length - 1);
    }

    int [] arrayLen4 = {0, 1, 2, 4, 5};

    void test2( int i) {
        Contract.Requires(i >= 0 && i <= this.arrayLen4.Length);
        if (i > 0) {
            int j = arrayLen4[i - 1];
        }
        int j2 = arrayLen4[1];
    }

    void callTest2(int x) {
        test2(0);
        test2(1);
        test2(2);
        test2(4);
        // :: error: (argument.type.incompatible)
        test2(5);
        test2(arrayLen4.Length);
    }
}
