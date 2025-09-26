// Tests that String.length() is supported in the same situations as array length

using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

class StringLength {
    void testMinLenSubtractPositive(String s) {
        Contract.Requires(s.Length >= 10);
        int i1 = s.Length - 9;
        if(TestHelper.nondet()) Contract.Assert(i1 > 0);
        int i0 = s.Length - 10;
        if(TestHelper.nondet()) Contract.Assert(i0 >= 0);
        // ::  error: (assignment.type.incompatible)
        int im1 = s.Length - 11;
        if(TestHelper.nondet()) Contract.Assert(im1 >= 0);
    }

    void testNewArraySameLen(String s) {
        int[] array = new int[s.Length]; // TODO
        if(TestHelper.nondet()) Contract.Assert(array.Length == s.Length);
        // ::  error: (assignment.type.incompatible)
        int[] array1 = new int[s.Length + 1];
        if(TestHelper.nondet()) Contract.Assert(array1.Length == s.Length);
    }

    void testStringAssignSameLen(String s, String r) {
        String t = s;
        if(TestHelper.nondet()) Contract.Assert(t.Length == s.Length);
        // ::  error: (assignment.type.incompatible)
        String tN = r;
        if(TestHelper.nondet()) Contract.Assert(tN.Length == s.Length);
    }

    void testStringLenEqualSameLen(String s, String r) {
        if (s.Length == r.Length) {
            String tN = r;
            if(TestHelper.nondet()) Contract.Assert(tN.Length == s.Length);
        }
    }

    void testStringEqualSameLen(String s, String r) {
        if (s == r) {
            String tN = r;
            if(TestHelper.nondet()) Contract.Assert(tN.Length == s.Length);
        }
    }

    void testOffsetRemoval(
            String s,
            String t,
            int i,
            int j,
            int k) {
        Contract.Requires(i + t.Length < s.Length);
        Contract.Requires(j < t.Length);

        int ij = i + j;

        if(TestHelper.nondet()) Contract.Assert(ij < s.Length);
        // ::  error: (assignment.type.incompatible)
        int ik = i + k;
        if(TestHelper.nondet()) Contract.Assert(ik < s.Length);
    }

    void testLengthDivide(String s) {
        Contract.Requires(s.Length >= 1);
        int i = s.Length / 2;
        if(TestHelper.nondet()) Contract.Assert(i >= 0 && i < s.Length);
    }

    void testAddDivide(String s, int i, int j) {
        Contract.Requires(s.Length >= 1);
        Contract.Requires(i >= 0 && i<s.Length);
        Contract.Requires(j >= 0 && j<s.Length);
        int ij = (i + j) / 2;
        if(TestHelper.nondet()) Contract.Assert(ij >= 0 && ij < s.Length);
    }

    void testRandomMultiply(String s, Random r) {
        Contract.Requires(s.Length >= 1);
#if JDK
        @LTLengthOf("s") int i = (int) (Math.random() * s.Length);
#endif
        int j = (int) (r.NextDouble() * s.Length);
        if(TestHelper.nondet()) Contract.Assert(j < s.Length);
    }

    void testNotEqualLength(String s, int i, int j) {
        Contract.Requires(i >= 0 && i <= s.Length);
        Contract.Requires(j >= 0 && j <= s.Length);
        if (i != s.Length) {
            int @in = i;
            if(TestHelper.nondet()) Contract.Assert(@in >= 0 && @in < s.Length);
            // ::  error: (assignment.type.incompatible)
            int jn = j;
            if(TestHelper.nondet()) Contract.Assert(jn >= 0 && jn < s.Length);
        }
    }

    void testLength(String s) {
        int i = s.Length;
        if(TestHelper.nondet()) Contract.Assert(i >= 0 && i <= s.Length);
        int j = s.Length - 1;
        if(TestHelper.nondet()) Contract.Assert(j < s.Length);
    }
}
