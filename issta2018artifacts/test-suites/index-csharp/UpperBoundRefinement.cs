using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

/*@SuppressWarnings("lowerbound")*/
public class UpperBoundRefinement {
    // If expression i has type @LTLengthOf(value = "f2", offset = "f1.length") int and expression
    // j is less than or equal to the length of f1, then the type of i + j is @LTLengthOf("f2")
    void test(int[] f1, int[] f2) {
        int i = (f2.Length - 1) - f1.Length;
        if(TestHelper.nondet()) Contract.Assert(i + f1.Length < f2.Length);
        int j = f1.Length - 1;
        if(TestHelper.nondet()) Contract.Assert(j < f1.Length);
        int x = i + j;
        if(TestHelper.nondet()) Contract.Assert(x < f2.Length);
        int y = i + f1.Length;
        if(TestHelper.nondet()) Contract.Assert(y < f2.Length);
    }

    void test2() {
        double[] f1 = new double[10];
        double[] f2 = new double[20];

        for (int j = 0; j < f2.Length; j++) {
            f2[j] = j;
        }
        for (int i = 0; i < f2.Length - f1.Length; i++) {
            // fill up f1 with elements of f2
            for (int j = 0; j < f1.Length; j++) {
                f1[j] = f2[i + j];
            }
        }
    }

    public void test3(double[] a, double[] sub) {
        int a_index_max = a.Length - sub.Length;
        // Has type @LTL(value={"a","sub"}, offset={"-1 + sub.length", "-1 + a.Length"})

        for (int i = 0; i <= a_index_max; i++) { // i has the same type as a_index_max
            for (int j = 0; j < sub.Length; j++) { // j is @LTL("sub")
                // i + j is safe here. Because j is LTL("sub"), it should count as ("-1 +
                // sub.length")
                double d = a[i + j];
            }
        }
    }
}
