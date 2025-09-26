using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

public class ConstantOffsets {
    void method1(int[] a, int offset, int x) {
        Contract.Requires(x - offset - 1 < a.Length);
    }

    void test1() {
        int offset = -4;
        int x = 4;
        int[] f1 = new int[x - offset];
        method1(f1, offset, x);
    }

    void method2(int[] a, int offset, int x) {
        Contract.Requires(x + offset - 1 < a.Length);
    }

    void test2() {
        int offset = 4;
        int x = 4;
        int[] f1 = new int[x + offset];
        method2(f1, offset, x);
    }
}
