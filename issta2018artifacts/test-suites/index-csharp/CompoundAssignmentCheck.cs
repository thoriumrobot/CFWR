using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

class CompoundAssignmentCheck {
    void test() {
        int a = 9;
        a += 5;
        a -= 2;
        int[] arr5 = new int[a]; // LBC shouldn't warn here.
    }
}
