using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

public sealed class Loops {
    public static bool flag = false;

    public void test1(int[] a, int offset, int offset2) {
        Contract.Requires(offset<a.Length);
        Contract.Requires(offset2 < a.Length);
        while (flag) {
            // :: error: (compound.assignment.type.incompatible)
            offset++;
            // :: error: (compound.assignment.type.incompatible)
            offset += 1;
            // :: error: (compound.assignment.type.incompatible)
            offset2 += offset;
        }
    }

    public void test2(int[] a, int[] array) {
        int offset = array.Length - 1;
        int offset2 = array.Length - 1;

        while (flag) {
            offset++;
            offset2 += offset;
        }
        // :: error: (assignment.type.incompatible)
        int x = offset;
        if(TestHelper.nondet()) Contract.Assert(x < array.Length);
        // :: error: (assignment.type.incompatible)
        int y = offset2;
        if(TestHelper.nondet()) Contract.Assert(y < array.Length);
    }

    public void test3(int[] a, int offset, int offset2) {
        Contract.Requires(offset < a.Length);
        Contract.Requires(offset2 < a.Length);
        while (flag) {
            offset--;
            // :: error: (compound.assignment.type.incompatible)
            offset2 -= offset;
        }
    }

    public void test4(int[] a, int offset, int offset2) {
        Contract.Requires(offset < a.Length);
        Contract.Requires(offset2 < a.Length);
        while (flag) {
            // :: error: (compound.assignment.type.incompatible)
            offset++;
            // :: error: (compound.assignment.type.incompatible)
            offset += 1;
            // :: error: (compound.assignment.type.incompatible)
            offset2 += offset;
        }
    }

    public void test4(int[] src) {
        int patternLength = src.Length;
        int[] optoSft = new int[patternLength];
        for (int i = patternLength; i > 0; i--) {}
    }

    public void test5(
            int[] a,
            int offset,
            int offset2) {

        Contract.Requires(offset + -1000 < a.Length);
        Contract.Requires(offset2 < a.Length);

        int otherOffset = offset;
        while (flag) {
            otherOffset += 1;
            // :: error: (compound.assignment.type.incompatible)
            offset++;
            // :: error: (compound.assignment.type.incompatible)
            offset += 1;
            // :: error: (compound.assignment.type.incompatible)
            offset2 += offset;
        }
        // :: error: (assignment.type.incompatible)
        int x = otherOffset;
        if(TestHelper.nondet()) Contract.Assert(otherOffset + -1000 < a.Length);
    }
}
