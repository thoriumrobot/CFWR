using System;
using System.Collections.Generic;
using System.Diagnostics.Contracts;
using System.Text;
using System.IO;

public class SwitchTest {

    public String findSlice_unordered(String[] vis) {
        switch (vis.Length) {
            case 1:
                int x = vis.Length;
                if(TestHelper.nondet()) Contract.Assert(x == 1);
                return vis[0];
            case 2:
                return vis[0] + vis[1];
            case 3:
                return vis[0] + vis[1] + vis[2];
            default:
                throw new Exception("Bad length " + vis.Length);
        }
    }
}
