using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

public class ParserOffsetTest {

    public void subtraction1(String[] a,  int i) {
        Contract.Requires(i >= 0 && i < a.Length);
        int length = a.Length;
        if (i >= length - 1 || a[i + 1] == null) {
            // body is irrelevant
        }
    }

    public void addition1(String[] a, int i) {
        Contract.Requires(i >= 0 && i < a.Length);
        int length = a.Length;
        if ((i + 1) >= length || a[i + 1] == null) {
            // body is irrelevant
        }
    }

    public void subtraction2(String[] a, int i) {
        Contract.Requires(i >= 0 && i < a.Length);
        if (i < a.Length - 1) {
            int j = i + 1;
            if(TestHelper.nondet()) Contract.Assert(j >= 0 && j < a.Length);
        }
    }

    public void addition2(String[] a, int i) {
        Contract.Requires(i >= 0 && i < a.Length);
        if ((i + 1) < a.Length) {
            int j = i + 1;
            if(TestHelper.nondet()) Contract.Assert(j >= 0 && j < a.Length);
        }
    }

    public void addition3(String[] a, int i) {
        Contract.Requires(i >= 0 && i < a.Length);
        if ((i + 5) < a.Length) {
            int j = i + 5;
            if(TestHelper.nondet()) Contract.Assert(j >= 0 && j < a.Length);
        }
    }

    /*@SuppressWarnings("lowerbound")*/
    public void subtraction3(String[] a, int k) {
        Contract.Requires(k >= 0);
        if (k - 5 < a.Length) {
            String s = a[k - 5];
            int j = k - 5;
            if(TestHelper.nondet()) Contract.Assert(j >= 0 && j < a.Length);
        }
    }

    /*@SuppressWarnings("lowerbound")*/
    public void subtraction4(String[] a, int i) {
        Contract.Requires(i >= 0 && i < a.Length);
        if (1 - i < a.Length) {
            // The error on this assignment is a false positive.
            // :: error: (assignment.type.incompatible)
            int j = 1 - i;
            if(TestHelper.nondet()) Contract.Assert(j >= 0 && j < a.Length);

            // :: error: (assignment.type.incompatible)
            int k = i;
            if(TestHelper.nondet()) Contract.Assert(k + 1 < a.Length);
        }
    }

    /*@SuppressWarnings("lowerbound")*/
    public void subtraction5(String[] a, int i) {
        if (1 - i < a.Length) {
            // :: error: (assignment.type.incompatible)
            int j = i;
            if(TestHelper.nondet()) Contract.Assert(j >= 0 && j < a.Length);
        }
    }

    /*@SuppressWarnings("lowerbound")*/
    public void subtraction6(String[] a, int i, int j) {
        if (i - j < a.Length - 1) {
            int k = i - j;
            if(TestHelper.nondet()) Contract.Assert(k >= 0 && k < a.Length);
            // :: error: (assignment.type.incompatible)
            int k1 = i;
            if(TestHelper.nondet()) Contract.Assert(k1 >= 0 && k1 < a.Length);
        }
    }

    public void multiplication1(String[] a, int i, int j) {
        Contract.Requires(j > 0);
        if ((i * j) < (a.Length + j)) {
            // :: error: (assignment.type.incompatible)
            int k = i;
            if(TestHelper.nondet()) Contract.Assert(k >= 0 && k < a.Length);
            // :: error: (assignment.type.incompatible)
            int k1 = j;
            if(TestHelper.nondet()) Contract.Assert(k1 >= 0 && k1 < a.Length);
        }
    }

    public void multiplication2(String [] a, int i, int j) {
        Contract.Requires(i == -2);
        Contract.Requires(j == 20);
        Contract.Requires(a.Length == 5);
        if ((i * j) < (a.Length - 20)) {
            int k1 = i;
            if(TestHelper.nondet()) Contract.Assert(k1 < a.Length);
            // :: error: (assignment.type.incompatible)
            int k2 = i;
            if(TestHelper.nondet()) Contract.Assert(k2 + 20 < a.Length);
            // :: error: (assignment.type.incompatible)
            int k3 = j;
            if(TestHelper.nondet()) Contract.Assert(k3 < a.Length);
        }
    }
}
