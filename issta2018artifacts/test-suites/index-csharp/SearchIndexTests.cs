using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

class SearchIndexTests {
    public void test(short[] a, short instant) {
        int i = Array.BinarySearch(a, instant);
        int z = i;
        if(TestHelper.nondet()) Contract.Assert(z >= - a.Length - 1 && z <= a.Length - 1);
        // :: error: (assignment.type.incompatible)
        int y = 7;
        if(TestHelper.nondet()) Contract.Assert(y >= - a.Length - 1 && y <= a.Length - 1);
        int x = i;
        if(TestHelper.nondet()) Contract.Assert(x < a.Length);
    }

    void test2(int[] a, int xyz) {
        Contract.Requires(xyz >= -a.Length -1 && xyz <= a.Length - 1);
        if (0 > xyz) {
            int w = xyz;
            if(TestHelper.nondet()) Contract.Assert(w >= - a.Length - 1 && w <= - 1);
            int y = ~xyz;
            if(TestHelper.nondet()) Contract.Assert(y >= 0);
            int z = ~xyz;
            if(TestHelper.nondet()) Contract.Assert(z <= a.Length);
        }
    }

    void test3(int[] a, int xyz) {
        Contract.Requires(xyz >= -a.Length -1 && xyz <= a.Length - 1);
        if (-1 >= xyz) {
            int w = xyz;
            if(TestHelper.nondet()) Contract.Assert(w >= - a.Length - 1 && w <= - 1);
            int y = ~xyz;
            if(TestHelper.nondet()) Contract.Assert(y >= 0);
            int z = ~xyz;
            if(TestHelper.nondet()) Contract.Assert(z <= a.Length);
        }
    }

    void test4(int[] a, int xyz) {
        Contract.Requires(xyz >= -a.Length -1 && xyz <= a.Length - 1);
        if (xyz < 0) {
            int w = xyz;
            if(TestHelper.nondet()) Contract.Assert(w >= - a.Length - 1 && w <= - 1);
            int y = ~xyz;
            if(TestHelper.nondet()) Contract.Assert(y >= 0);
            int z = ~xyz;
            if(TestHelper.nondet()) Contract.Assert(z <= a.Length);
        }
    }

    void test5(int[] a, int xyz) {
        Contract.Requires(xyz >= -a.Length -1 && xyz <= a.Length - 1);
        if (xyz <= -1) {
            int w = xyz;
            if(TestHelper.nondet()) Contract.Assert(w >= - a.Length - 1 && w <= - 1);
            int y = ~xyz;
            if(TestHelper.nondet()) Contract.Assert(y >= 0);
            int z = ~xyz;
            if(TestHelper.nondet()) Contract.Assert(z <= a.Length);
        }
    }

    void subtyping1(
            int x, int y, int[] a, int[] b) {

        Contract.Requires(x >= -a.Length -1 && x <= a.Length - 1 && x >= -b.Length -1 && x <= b.Length - 1);
        Contract.Requires(y >= -a.Length -1 && y <= -1);

        // :: error: (assignment.type.incompatible)
        int z = y;
        if(TestHelper.nondet()) Contract.Assert(z >= -a.Length - 1 && z <= a.Length - 1 && z >= -b.Length - 1 && z <= b.Length - 1);
        int w = y;
        if(TestHelper.nondet()) Contract.Assert(w >= - a.Length - 1 && w <= a.Length - 1);
        int p = x;
        if(TestHelper.nondet()) Contract.Assert(p >= - b.Length - 1 && p <= b.Length - 1);
        // :: error: (assignment.type.incompatible)
        int q = x;
        if(TestHelper.nondet()) Contract.Assert(z >= -a.Length - 1 && z <= - 1 && z >= -b.Length - 1 && z <= - 1);
    }
}
