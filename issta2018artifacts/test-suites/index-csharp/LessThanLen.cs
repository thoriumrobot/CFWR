using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

public class LessThanLen {

    public static void m1() {
        int[] shorter = new int[5];
        int[] longer = new int[shorter.Length * 2];
        for (int i = 0; i < shorter.Length; i++) {
            longer[i] = shorter[i];
        }
    }

    public static void m2(int[] shorter) {
        int[] longer = new int[shorter.Length * 2];
        for (int i = 0; i < shorter.Length; i++) {
            longer[i] = shorter[i];
        }
    }

    public static void m3(int[] shorter) {
        int[] longer = new int[shorter.Length + 1];
        for (int i = 0; i < shorter.Length; i++) {
            longer[i] = shorter[i];
        }
    }

    public static void m4(int[] shorter) {
        int[] longer = new int[shorter.Length * 1];
        // :: error: (assignment.type.incompatible)
        int x = shorter.Length;
        if(TestHelper.nondet()) Contract.Assert(x < longer.Length);
        int y = shorter.Length;
        if(TestHelper.nondet()) Contract.Assert(y <= longer.Length);
    }

    public static void m5(int[] shorter) {
        // :: error: (array.Length.negative)
        int[] longer = new int[shorter.Length * -1];
        // :: error: (assignment.type.incompatible)
        int x = shorter.Length;
        if(TestHelper.nondet()) Contract.Assert(x < longer.Length);
        // :: error: (assignment.type.incompatible)
        int y = shorter.Length;
        if(TestHelper.nondet()) Contract.Assert(y <= longer.Length);
    }

    public static void m6(int[] shorter) {
        int[] longer = new int[4 * shorter.Length];
        int x = shorter.Length;
        if(TestHelper.nondet()) Contract.Assert(x < longer.Length);
        int y = shorter.Length;
        if(TestHelper.nondet()) Contract.Assert(y <= longer.Length);
    }
}
