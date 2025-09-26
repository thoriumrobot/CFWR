using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

public class RefinementLT {

    void test_backwards(int a, int j, int s) {
        /** backwards less than */
        // :: error: (assignment.type.incompatible)
        int aa = a;
        if(TestHelper.nondet()) Contract.Assert(aa >= 0);
        if (-1 < a) {
            int b = a;
            if(TestHelper.nondet()) Contract.Assert(b >= 0);
        } else {
            // :: error: (assignment.type.incompatible)
            int c = a;
            if(TestHelper.nondet()) Contract.Assert(c >= 0);
        }

        if (0 < j) {
            int k = j;
            if(TestHelper.nondet()) Contract.Assert(k > 0);
        } else {
            // :: error: (assignment.type.incompatible)
            int l = j;
            if(TestHelper.nondet()) Contract.Assert(l > 0);
        }

        if (1 < s) {
            int t = s;
            if(TestHelper.nondet()) Contract.Assert(t > 0);
        } else {
            // :: error: (assignment.type.incompatible)
            int u = s;
            if(TestHelper.nondet()) Contract.Assert(u > 0);
        }
    }

    void test_forwards(int a, int j, int s) {
        /** forwards less than */
        // :: error: (assignment.type.incompatible)
        int aa = a;
        if(TestHelper.nondet()) Contract.Assert(aa >= 0);
        if (a < -1) {
            // :: error: (assignment.type.incompatible)
            int b = a;
            if(TestHelper.nondet()) Contract.Assert(b >= -1);
        } else {
            int c = a;
            if(TestHelper.nondet()) Contract.Assert(c >= -1);
        }

        if (j < 0) {
            // :: error: (assignment.type.incompatible)
            int k = j;
            if(TestHelper.nondet()) Contract.Assert(k >= 0);
        } else {
            int l = j;
            if(TestHelper.nondet()) Contract.Assert(l >= 0);
        }

        if (s < 1) {
            // :: error: (assignment.type.incompatible)
            int t = s;
            if(TestHelper.nondet()) Contract.Assert(t > 0);
        } else {
            int u = s;
            if(TestHelper.nondet()) Contract.Assert(u > 0);
        }
    }
}
// a comment
